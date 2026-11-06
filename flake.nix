{
  description = "Dev shell for vr180-convert (PyTorch, OpenCV, CUDA).";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        system = system;
        config.allowUnfree = true;
      };
      cudatookit-with-cudart-to-lib64 = pkgs.symlinkJoin {
        name = "cudatoolkit";
        paths = with pkgs.cudaPackages; [
          cudatoolkit
          (pkgs.lib.getStatic cuda_cudart)
        ];
        postBuild = ''
          ln -s $out/lib $out/lib64
        '';
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          # OpenCV dependencies
          xorg.libxcb
          xorg.libX11
          xorg.libXext
          xorg.libXrender
          xorg.libXi
          xorg.libXtst
          glib
          pkg-config
        ];

        shellHook = ''
          # Required for both PyTorch and Numba to find CUDA
          export CUDA_PATH=${cudatookit-with-cudart-to-lib64}

          # Required for both PyTorch and Numba, adds necessary paths for dynamic linking
          export LD_LIBRARY_PATH=${
            pkgs.lib.makeLibraryPath ([
              "/run/opengl-driver" # Needed to find libGL.so, required by both PyTorch and Numba
            ] ++ (with pkgs; [
              xorg.libxcb
              xorg.libX11
              xorg.libXext
              xorg.libXrender
              xorg.libXi
              xorg.libXtst
              glib
            ]))
          }:$LD_LIBRARY_PATH
        '';
      };
    };
}
