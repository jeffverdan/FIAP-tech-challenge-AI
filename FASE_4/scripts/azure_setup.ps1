# Cria (ou reaproveita) os recursos Azure da FASE 4 na assinatura ativa do Azure CLI
# e grava as chaves no FASE_4/.env.  Para TROCAR DE CONTA: `az login` na conta nova e rode de novo.
#
# Uso (PowerShell, dentro de FASE_4):
#   az login
#   .\scripts\azure_setup.ps1                     # F0 (gratuito) em eastus
#   .\scripts\azure_setup.ps1 -Sku S0 -Regiao brazilsouth
#
# Recursos criados no grupo rg-aurora-fase4:
#   * Speech Services (Speech to Text)
#   * Language / Text Analytics (sentimento, frases-chave, Text Analytics for Health)
param(
    [string]$Regiao = "eastus",
    [string]$Grupo = "rg-aurora-fase4",
    [ValidateSet("F0", "S0")][string]$Sku = "F0"
)
$ErrorActionPreference = "Stop"

if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    Write-Error "Azure CLI não encontrado. Instale: winget install -e --id Microsoft.AzureCLI"
}
$conta = az account show --query "{nome:name, id:id, usuario:user.name}" -o json | ConvertFrom-Json
Write-Host "Assinatura ativa: $($conta.nome) ($($conta.id)) — $($conta.usuario)" -ForegroundColor Cyan

# sufixo estável por assinatura (nomes de recurso precisam ser únicos globalmente)
$sufixo = $conta.id.Substring(0, 6)
$speech = "aurora-speech-$sufixo"
$language = "aurora-language-$sufixo"

az group create -n $Grupo -l $Regiao -o none

# assinaturas novas não vêm com o provedor de Cognitive Services registrado
$estado = az provider show -n Microsoft.CognitiveServices --query registrationState -o tsv
if ($estado -ne "Registered") {
    Write-Host "  registrando o provedor Microsoft.CognitiveServices (1–3 min, só na 1ª vez)…"
    az provider register -n Microsoft.CognitiveServices --wait -o none
}

function Garantir-Recurso($nome, $kind, $extra) {
    $existe = az cognitiveservices account list -g $Grupo --query "[?name=='$nome'] | length(@)" -o tsv
    if ($existe -eq "1") { Write-Host "  $nome já existe — reaproveitando"; return }
    Write-Host "  criando $nome ($kind, $Sku)…"
    $cmd = "az cognitiveservices account create -n $nome -g $Grupo -l $Regiao --kind $kind --sku $Sku --yes -o none $extra"
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Falhou ao criar $nome. Se o erro acima for de cota do F0 (1 recurso gratuito por tipo por assinatura), rode de novo com -Sku S0."
        throw "falha ao criar $nome"
    }
}

Garantir-Recurso $speech "SpeechServices" ""
Garantir-Recurso $language "TextAnalytics" "--custom-domain $language"

$speechKey = az cognitiveservices account keys list -n $speech -g $Grupo --query key1 -o tsv
$langKey = az cognitiveservices account keys list -n $language -g $Grupo --query key1 -o tsv
$langEndpoint = az cognitiveservices account show -n $language -g $Grupo --query properties.endpoint -o tsv

# grava/atualiza o .env (mantém as outras variáveis)
$envPath = Join-Path (Split-Path $PSScriptRoot -Parent) ".env"
if (-not (Test-Path $envPath)) { Copy-Item (Join-Path (Split-Path $PSScriptRoot -Parent) ".env.example") $envPath }
$valores = @{
    "AZURE_SPEECH_KEY" = $speechKey; "AZURE_SPEECH_REGION" = $Regiao;
    "AZURE_LANGUAGE_KEY" = $langKey; "AZURE_LANGUAGE_ENDPOINT" = $langEndpoint
}
$linhas = Get-Content $envPath
foreach ($k in $valores.Keys) {
    $achou = $false
    $linhas = $linhas | ForEach-Object { if ($_ -match "^$k=") { $achou = $true; "$k=$($valores[$k])" } else { $_ } }
    if (-not $achou) { $linhas += "$k=$($valores[$k])" }
}
Set-Content -Path $envPath -Value $linhas -Encoding utf8
Write-Host "`n.env atualizado. Validando…" -ForegroundColor Green
python (Join-Path $PSScriptRoot "check_azure.py")
