{
  description = "tt-spatter with OpenMP support";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        # Use LLVM's OpenMP on Darwin, GCC's on Linux
        openmp = if pkgs.stdenv.isDarwin
          then pkgs.llvmPackages.openmp
          else null;
        
        stdenv = if pkgs.stdenv.isDarwin
          then pkgs.llvmPackages.stdenv
          else pkgs.stdenv;
      in
      {
        devShells.default = stdenv.mkDerivation {
          name = "tt-spatter-dev";
          
          buildInputs = with pkgs; [
            cmake
            ninja
            pkg-config
            zlib
          ] ++ (if stdenv.isDarwin then [
            openmp
          ] else [
            gcc
          ]);
          
          shellHook = ''
            echo "tt-spatter development environment"
            echo "OpenMP is available"
            ${if stdenv.isDarwin then ''
              export OpenMP_ROOT="${openmp}"
              export CMAKE_PREFIX_PATH="${openmp}:$CMAKE_PREFIX_PATH"
            '' else ''
              echo "Using GCC with built-in OpenMP support"
            ''}
          '';
        };
        
        packages.default = stdenv.mkDerivation {
          pname = "tt-spatter";
          version = "0.1.0";
          
          src = ./.;
          
          nativeBuildInputs = with pkgs; [
            cmake
            ninja
            pkg-config
          ];
          
          buildInputs = if stdenv.isDarwin then [ openmp ] else [];
          
          cmakeFlags = if stdenv.isDarwin then [
            "-DOpenMP_ROOT=${openmp}"
          ] else [];
        };
      });
}