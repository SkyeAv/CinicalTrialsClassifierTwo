{ stdenvNoCC
, fetchurl
, lib
,
}:
let
  srcs = [
    (fetchurl {
      url = "https://huggingface.co/dmis-lab/biobert-v1.1/resolve/main/config.json";
      sha256 = "sha256-1Qbo0M/VIjRyZ46JfLKU9nZdLkaV05RFpuSGz4U+Yug=";
    })
    (fetchurl {
      url = "https://huggingface.co/dmis-lab/biobert-v1.1/resolve/main/pytorch_model.bin";
      sha256 = "sha256-jofLJ9iJGGxEL+/yvqU/jM4sX8JNS8iL8IzNOCGxONw=";
    })
    (fetchurl {
      url = "https://huggingface.co/dmis-lab/biobert-v1.1/resolve/main/vocab.txt";
      sha256 = "sha256-7qqYdbI7BLTFTvdZ0D250boVVIOPj7JsXZb6VR35PQI=";
    })
    (fetchurl {
      url = "https://huggingface.co/dmis-lab/biobert-v1.1/resolve/main/tokenizer_config.json";
      sha256 = "sha256-D20T5vTab54k8iraa8O+VxEj2FjXwMBainzVWpwjwug=";
    })
    (fetchurl {
      url = "https://huggingface.co/dmis-lab/biobert-v1.1/resolve/main/special_tokens_map.json";
      sha256 = "sha256-MD30WgNgnk6tBLw9wVNtCrGbU1jbaFtvPaEj0F7CAOM=";
    })
  ];
in
stdenvNoCC.mkDerivation {
  pname = "BioBert";
  version = "main";
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
