#!/usr/bin/env bash
# Versão bash de scripts/azure_setup.ps1 (Linux/macOS/WSL).
#   az login && ./scripts/azure_setup.sh [regiao] [F0|S0]
set -euo pipefail
REGIAO="${1:-eastus}"
SKU="${2:-F0}"
GRUPO="rg-aurora-fase4"
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"

ID=$(az account show --query id -o tsv)
echo "Assinatura ativa: $(az account show --query name -o tsv) ($ID)"
SUF="${ID:0:6}"
SPEECH="aurora-speech-$SUF"
LANG="aurora-language-$SUF"

az group create -n "$GRUPO" -l "$REGIAO" -o none
if [ "$(az provider show -n Microsoft.CognitiveServices --query registrationState -o tsv)" != "Registered" ]; then
  echo "  registrando o provedor Microsoft.CognitiveServices (1–3 min, só na 1ª vez)…"
  az provider register -n Microsoft.CognitiveServices --wait -o none
fi
criar() {  # nome kind [extra]
  if [ "$(az cognitiveservices account list -g "$GRUPO" --query "[?name=='$1'] | length(@)" -o tsv)" = "1" ]; then
    echo "  $1 já existe — reaproveitando"; return; fi
  echo "  criando $1 ($2, $SKU)…"
  az cognitiveservices account create -n "$1" -g "$GRUPO" -l "$REGIAO" --kind "$2" --sku "$SKU" --yes -o none ${3:-} \
    || { echo "Falhou. Se for cota do F0 (1 recurso gratuito por tipo), rode de novo com S0."; exit 1; }
}
criar "$SPEECH" SpeechServices
criar "$LANG" TextAnalytics "--custom-domain $LANG"

SK=$(az cognitiveservices account keys list -n "$SPEECH" -g "$GRUPO" --query key1 -o tsv)
LK=$(az cognitiveservices account keys list -n "$LANG" -g "$GRUPO" --query key1 -o tsv)
LE=$(az cognitiveservices account show -n "$LANG" -g "$GRUPO" --query properties.endpoint -o tsv)

ENV="$RAIZ/.env"
[ -f "$ENV" ] || cp "$RAIZ/.env.example" "$ENV"
definir() { if grep -q "^$1=" "$ENV"; then sed -i.bak "s|^$1=.*|$1=$2|" "$ENV" && rm -f "$ENV.bak"; else echo "$1=$2" >> "$ENV"; fi; }
definir AZURE_SPEECH_KEY "$SK"
definir AZURE_SPEECH_REGION "$REGIAO"
definir AZURE_LANGUAGE_KEY "$LK"
definir AZURE_LANGUAGE_ENDPOINT "$LE"
echo ".env atualizado. Validando…"
python "$RAIZ/scripts/check_azure.py"
