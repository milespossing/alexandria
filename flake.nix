{
  description = "Alexandria — semantic code search via MCP, backed by vector embeddings";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    let
      # NixOS module is system-independent
      nixosModule = { config, lib, pkgs, ... }:
        let
          cfg = config.services.alexandria;
          alexPkg = self.packages.${pkgs.system}.default;

          # Common environment for all Alexandria services
          alexandriaEnv = {
            QDRANT_URL = "http://localhost:${toString cfg.qdrant.port}";
            OLLAMA_HOST = "http://localhost:11434";
          };

          # Generate a concrete indexer service for each entry in cfg.indexes
          indexerServices = lib.mapAttrs' (name: indexCfg:
            lib.nameValuePair "alexandria-indexer-${name}" {
              description = "Alexandria indexer for ${name}";
              after = [ "alexandria-setup.service" ];
              requires = [ "alexandria-setup.service" ];
              environment = alexandriaEnv;
              serviceConfig = {
                Type = "oneshot";
                ExecStart = "${alexPkg}/bin/alex index --context ${name} ${toString indexCfg.path}";
              };
            }
          ) cfg.indexes;

          # Generate a matching timer for each indexer
          indexerTimers = lib.mapAttrs' (name: _indexCfg:
            lib.nameValuePair "alexandria-indexer-${name}" {
              description = "Periodic re-index for ${name}";
              wantedBy = [ "timers.target" ];
              timerConfig = {
                OnCalendar = cfg.reindexSchedule;
                Persistent = true;
                RandomizedDelaySec = "5m";
              };
            }
          ) cfg.indexes;
        in
        {
          options.services.alexandria = {
            enable = lib.mkEnableOption "Alexandria semantic code search";

            qdrant = {
              port = lib.mkOption {
                type = lib.types.port;
                default = 6333;
                description = "Port for Qdrant HTTP API.";
              };

              grpcPort = lib.mkOption {
                type = lib.types.port;
                default = 6334;
                description = "Port for Qdrant gRPC API.";
              };
            };

            ollama = {
              model = lib.mkOption {
                type = lib.types.str;
                default = "nomic-embed-text";
                description = "Ollama embedding model to pull and use.";
              };
            };

            reindexSchedule = lib.mkOption {
              type = lib.types.str;
              default = "daily";
              description = "systemd calendar expression for periodic re-indexing.";
              example = "hourly";
            };

            indexes = lib.mkOption {
              type = lib.types.attrsOf (lib.types.submodule {
                options = {
                  path = lib.mkOption {
                    type = lib.types.path;
                    description = "Path to the codebase to index.";
                  };
                };
              });
              default = { };
              description = "Codebases to index. Attribute name becomes the context name.";
              example = {
                myproject = { path = "/home/user/src/myproject"; };
              };
            };
          };

          config = lib.mkIf cfg.enable {
            # Put alex on the system PATH
            environment.systemPackages = [ alexPkg ];

            # Qdrant via upstream NixOS module (native systemd service)
            services.qdrant = {
              enable = true;
              settings = {
                service = {
                  host = "127.0.0.1";
                  http_port = cfg.qdrant.port;
                  grpc_port = cfg.qdrant.grpcPort;
                };
              };
            };

            # Ollama (upstream NixOS module)
            services.ollama.enable = true;

            # Oneshot: wait for Qdrant + Ollama, then pull the embedding model
            # Uses `alex setup` which verifies both services and pulls the model
            # Plus concrete indexer services (one per configured index)
            systemd.services = {
              alexandria-setup = {
                description = "Alexandria setup — pull Ollama embedding model and verify Qdrant";
                after = [ "ollama.service" "qdrant.service" ];
                requires = [ "ollama.service" "qdrant.service" ];
                wantedBy = [ "multi-user.target" ];
                environment = alexandriaEnv;
                serviceConfig = {
                  Type = "oneshot";
                  RemainAfterExit = true;
                  ExecStart = "${alexPkg}/bin/alex setup";
                };
              };
            } // indexerServices;

            # Timers for periodic re-indexing
            systemd.timers = indexerTimers;
          };
        };
    in
    (flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        python = pkgs.python3;

        # All Python dependencies from nixpkgs
        pythonDeps = ps: [
          ps.qdrant-client
          ps.tree-sitter
          ps.tree-sitter-language-pack
          ps.ollama
          ps.mcp
          ps.click
          ps.rich
          ps.tqdm
          ps.pathspec
          ps.pyyaml
        ];

        # Dev-only Python dependencies
        pythonDevDeps = ps: [
          ps.pytest
          ps.pytest-asyncio
          ps.black
          ps.ruff
          ps.mypy
        ];

        # The Alexandria Python package, built with nixpkgs infrastructure
        alexandriaPkg = python.pkgs.buildPythonApplication {
          pname = "alexandria";
          version = "0.1.0";
          pyproject = true;

          src = ./.;

          build-system = [ python.pkgs.setuptools ];

          dependencies = pythonDeps python.pkgs;

          nativeCheckInputs = [
            python.pkgs.pytest
          ];

          # fd is used at runtime for file discovery (gitignore-aware)
          makeWrapperArgs = [
            "--prefix PATH : ${pkgs.lib.makeBinPath [ pkgs.fd ]}"
          ];

          meta = {
            description = "Semantic code search via MCP, backed by vector embeddings";
            mainProgram = "alex";
          };
        };

        # Python environment for the dev shell (editable installs happen manually)
        pythonEnv = python.withPackages (ps:
          (pythonDeps ps) ++ (pythonDevDeps ps)
        );

      in
      {
        packages = {
          default = alexandriaPkg;
          alexandria = alexandriaPkg;

          docker = pkgs.dockerTools.buildLayeredImage {
            name = "alexandria";
            tag = "latest";
            contents = [
              alexandriaPkg
              pkgs.coreutils
              pkgs.cacert
            ];
            config = {
              Entrypoint = [ "${alexandriaPkg}/bin/alexandria" ];
              Env = [
                "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
              ];
            };
          };
        };

        apps.default = {
          type = "app";
          program = "${alexandriaPkg}/bin/alex";
        };

        apps.alex = {
          type = "app";
          program = "${alexandriaPkg}/bin/alex";
        };

        devShells.default = pkgs.mkShell {
          name = "alexandria-dev";

          packages = [
            pythonEnv
            pkgs.fd
            pkgs.qdrant

            # For running Ollama locally during development
            pkgs.ollama
          ];

          env = {
            # Point to a local Qdrant (default port)
            QDRANT_URL = "http://localhost:6333";
            OLLAMA_HOST = "http://localhost:11434";
            ALEXANDRIA_DEV = "1";


          };

          shellHook = ''
            # Make the src-layout package importable without pip install -e.
            # Uses the live working tree so edits are reflected immediately.
            export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"

            echo "Alexandria dev shell"
            echo "  Python:  $(python3 --version)"
            echo "  Qdrant:  $(qdrant --version 2>/dev/null || echo 'available')"
            echo "  Ollama:  $(ollama --version 2>/dev/null || echo 'available')"
            echo ""
            echo "Run 'qdrant' in a separate terminal to start the vector DB."
            echo "Run 'ollama serve' in a separate terminal, then 'ollama pull nomic-embed-text'."
            echo ""
            echo "To run Alexandria in dev mode:"
            echo "  python -m alexandria.cli --help"
          '';
        };
      }
    )) // {
      # System-independent outputs
      nixosModules.default = nixosModule;
    };
}
