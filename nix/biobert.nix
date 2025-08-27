{
  stdenvNoCC,
  fetchurl,
  lib,
}:
stdenvNoCC.mkDerivation {
  pname = "BioBert";
  version = "main";
  srcs = [
    (fetchurl {
      url = "https://huggingface.co/dmis-lab/biobert-v1.1/resolve/main/config.json";
      sha256 = lib.fakeSha256;
    })
    (fetchurl {
      url = "https://huggingface.co/dmis-lab/biobert-v1.1/resolve/main/pytorch_model.bin";
      sha256 = lib.fakeSha256;
    })
    (fetchurl {
      url = "https://huggingface.co/dmis-lab/biobert-v1.1/resolve/main/vocab.txt";
      sha256 = lib.fakeSha256;
    })
    (fetchurl {
      url = "https://huggingface.co/dmis-lab/biobert-v1.1/resolve/main/tokenizer_config.json";
      sha256 = lib.fakeSha256;
    })
    (fetchurl {
      url = "https://huggingface.co/dmis-lab/biobert-v1.1/resolve/main/special_tokens_map.json";
      sha256 = lib.fakeSha256;
    })
  ];
  dontUnpack = true;
  installPhase = ''
    runHook preInstall
    mkdir -p $out
    cp ${builtins.elemAt srcs 0} $out/config.json
    cp ${builtins.elemAt srcs 1} $out/pytorch_model.bin
    cp ${builtins.elemAt srcs 2} $out/vocab.txt
    cp ${builtins.elemAt srcs 3} $out/tokenizer_config.json
    cp ${builtins.elemAt srcs 4} $out/special_tokens_map.json
    runHook postInstall
  '';
}
