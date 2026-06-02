# LLM Eval CI/CD Pipeline

> Zautomatyzowany pipeline ewaluacyjny, który testuje outputy LLM przy każdej zmianie
> promptów, modeli lub bazy wiedzy RAG — tak jak testy jednostkowe uruchamiają się przy
> zmianach w kodzie.

<!-- TODO (Krok 11): badge'e CI, coverage, licencja -->

## Spis treści

- [Problem](#problem)
- [Architektura](#architektura) <!-- TODO: diagram mermaid (Krok 11) -->
- [Mierzone metryki](#mierzone-metryki)
- [Szybki start](#szybki-start)
- [Struktura projektu](#struktura-projektu)
- [Screenshoty](#screenshoty) <!-- TODO: zrzuty dashboardu (Krok 11) -->
- [Czego się nauczyłam](#czego-się-nauczyłam) <!-- TODO (Krok 11) -->

## Problem

Outputy LLM potrafią cicho się pogorszyć (regresja), gdy zmienimy prompt, podmienimy model
albo zaktualizujemy bazę wiedzy RAG. Nie widać tego w klasycznych testach jednostkowych.
Ten projekt traktuje jakość odpowiedzi LLM jak coś, co da się **zmierzyć i bramkować w CI**.

## Architektura

<!-- TODO (Krok 11): diagram mermaid -->
Golden dataset → Pipeline (RAG + LLM) → Metryki → Bramka SLA (gate) → Storage (SQLite) → Dashboard.

## Mierzone metryki

- **Hallucination rate** — odsetek odpowiedzi z treścią niepopartą źródłami (LLM-as-judge).
- **Answer relevancy** — czy odpowiedź faktycznie odpowiada na pytanie.
- **Faithfulness** — wierność odpowiedzi wobec dostarczonego kontekstu.
- **Latency p50 / p95** — opóźnienie (percentyle, nie średnia).
- **Koszt / zapytanie** — liczba tokenów × cennik modelu.

## Szybki start

```bash
# 1. Zależności (wymaga zainstalowanego uv: https://docs.astral.sh/uv/)
uv sync --extra dev

# 2. Konfiguracja środowiska
cp .env.example .env   # tryb domyślny "mock" działa bez klucza API

# (kolejne komendy — seed danych, uruchomienie evalu, dashboard — pojawią się w sekcjach poniżej w miarę budowy projektu)
```

## Struktura projektu

```
config/    # progi SLA (thresholds.yaml) i definicje modeli (models.yaml)
data/      # golden_dataset.jsonl + dokumenty bazy wiedzy RAG
src/
  pipeline/  # RAG, klient LLM, prompty
  eval/      # metryki, runner, bramka SLA
  storage/   # zapis metryk w SQLite
  dashboard/ # dashboard Streamlit
scripts/   # seed danych, ingest bazy wiedzy
tests/     # testy pytest
```

## Screenshoty

<!-- TODO (Krok 11) -->

## Czego się nauczyłam

<!-- TODO (Krok 11) -->