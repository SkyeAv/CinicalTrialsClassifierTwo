{
  inputs,
  self,
  ...
}: {
  perSystem = {
    system,
    lib,
    ...
  }: let
    pkgs = nixpkgs;
    moduleName = "CTClassifier2";
    seed = "87";
  in {
    py = pkgs.python313Packages;
    BioBert = pkgs.callPackage ./nix/biobert.nix {};
    PyTorchFrame = py.buildPythonWheel rec {
      pname = "pytorch-frame";
      version = "0.2.5";
      src = fetchurl {
        url = "https://files.pythonhosted.org/packages/2a/da/75804267b2bd9839bc44ba60cadde60bdcb50261a8cf448d54a81ce04334/pytorch_frame-0.2.5-py3-none-any.whl";
        sha256 = lib.fakeSha256;
      };
    };
    Module = buildPythonApplication {
      pname = "${moduleName}";
      version = "2.0.0";
      src = ./.;
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
        loguru
        pandas
        duckdb
        polars
        fsspec
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
      pythonImportsCheck = [
        "${moduleName}"
      ];
    };
    default = self.packages.${system}.Module;
  };
}
