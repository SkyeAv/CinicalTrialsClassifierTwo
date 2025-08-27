{
  description = "CTClassifier";
  inputs = import ./nix/imports.nix;
  outputs = inputs @ {
    self,
    systems,
    nixpkgs,
    flake-parts,
    ...
  }:
    flake-parts.lib.mkFlake {
      inherit inputs;
    } {
      systems = import inputs.systems;
      imports = [
        ./nix/app.nix
      ];
    };
}
