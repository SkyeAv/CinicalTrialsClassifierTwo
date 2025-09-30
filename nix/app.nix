{ inputs
, ...
}: {
  perSystem =
    { system
    , config
    , ...
    }:
    let
      moduleName = "CTClassifier2";
      seed = "87";
      pkgs = import inputs.nixpkgs { inherit system; };
      lib = pkgs.lib;
      py = pkgs.python313Packages;
      BioBert = pkgs.callPackage ./biobert.nix { };
      PyTorchFrame = py.buildPythonPackage rec {
        pname = "pytorch-frame";
        version = "0.2.5";
        format = "wheel";
        src = pkgs.fetchurl {
          url = "https://files.pythonhosted.org/packages/2a/da/75804267b2bd9839bc44ba60cadde60bdcb50261a8cf448d54a81ce04334/pytorch_frame-0.2.5-py3-none-any.whl";
          sha256 = "sha256-QPUcqK1yBYYO1YSCC8PWOsT/2px0O6SgsvU1H30iyUk=";
        };
      };
    in
    {
      devShells.default = pkgs.mkShell {
        packages = [
          config.packages.app
          py.flake8
        ];
      };
      packages.app = py.buildPythonApplication {
        pname = "${moduleName}";
        version = "2.0.0";
        src = ../.;
        pyproject = true;
        build-system = with py; [
          setuptools
        ];
        dependencies = with py; [
          PyTorchFrame
          transformers
          ruamel-yaml
          accelerate
          pydantic
          pyarrow
          BioBert
          joblib
          loguru
          pandas
          duckdb
          polars
          fsspec
          typer
          torch
          numpy
        ];
        nativeBuildInputs = with pkgs; [
          makeWrapper
        ];
        makeWrapperArgs = [
          "--set BIOBERT_DIR ${BioBert}"
          "--set TRANSFORMERS_OFFLINE 1"
          "--set TOKENIZERS_PARALLELISM true"
          "--set PYTHONHASHSEED ${seed}"
        ];
      };
    };
}
