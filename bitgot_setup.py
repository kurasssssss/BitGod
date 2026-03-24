"""
BITGOT — KONFIGURACJA BITGET API
=================================
Uruchom ten skrypt raz aby zweryfikować połączenie z Bitget.

BEZPIECZEŃSTWO:
  Klucze API ustaw TYLKO jako zmienne środowiskowe (Replit Secrets).
  NIGDY nie wpisuj ich bezpośrednio do kodu.

REPLIT SECRETS (zakładka Secrets w lewym panelu):
  BITGET_KEY  = bg_e4368d62c5c7b75c93779ed0ba9b1248
  BITGET_SECRET = 20a48c67a3960a49645ff6323210fa0e817cd73ef11fcde219bea3aa05810c97
  BITGET_PASS = <Twoje hasło API — wpisałeś przy tworzeniu klucza>

UPRAWNIENIA (widoczne na screenie — idealne dla BITGOT):
  ✅ Zlecenie futures (read/write)
  ✅ Otwarte pozycje (read/write)
  ✅ Handel spot (read/write)
  ✅ Spot margin (read/write)
  ✅ Copy trading (read/write)
"""

import os
import asyncio
import sys

# ── Odczyt zmiennych środowiskowych ───────────────────────────────────────────
BITGET_KEY    = os.getenv("BITGET_KEY",    "")
BITGET_SECRET = os.getenv("BITGET_SECRET", "")
BITGET_PASS   = os.getenv("BITGET_PASS",   "")

def check_env():
    missing = []
    if not BITGET_KEY:    missing.append("BITGET_KEY")
    if not BITGET_SECRET: missing.append("BITGET_SECRET")
    if not BITGET_PASS:   missing.append("BITGET_PASS")
    if missing:
        print(f"\n❌ Brak zmiennych środowiskowych: {missing}")
        print("\nJak ustawić w Replit:")
        print("  1. Kliknij ikonę 🔒 'Secrets' w lewym panelu")
        print("  2. Dodaj każdy klucz osobno:")
        print("     BITGET_KEY     → bg_e4368d62c5c7b75c93779ed0ba9b1248")
        print("     BITGET_SECRET  → 20a48c67a3960a49645ff6...c97")
        print("     BITGET_PASS    → [Twoje hasło API z Bitget]")
        print("\n⚠️  Po ustawieniu unieważnij stary klucz i wygeneruj nowy!")
        return False
    print("✅ Zmienne środowiskowe załadowane")
    print(f"   KEY:    {BITGET_KEY[:8]}...{BITGET_KEY[-4:]}")
    print(f"   SECRET: {BITGET_SECRET[:8]}...{BITGET_SECRET[-4:]}")
    print(f"   PASS:   {'*' * len(BITGET_PASS)}")
    return True

async def test_connection():
    """Testuj połączenie z Bitget API."""
    print("\n📡 Testuję połączenie z Bitget...")
    try:
        import ccxt.async_support as ccxt
    except ImportError:
        print("❌ ccxt nie zainstalowane. Uruchom: pip install ccxt")
        return False

    exchange = ccxt.bitget({
        "apiKey":   BITGET_KEY,
        "secret":   BITGET_SECRET,
        "password": BITGET_PASS,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",
        },
    })

    try:
        # Test 1: Publiczne dane (bez autentykacji)
        print("  [1/4] Ładowanie rynków...")
        markets = await exchange.load_markets()
        futures = [s for s in markets if ":USDT" in s and markets[s].get("type") == "swap"]
        print(f"  ✅ {len(markets)} rynków łącznie, {len(futures)} USDT futures")

        # Test 2: Autentykacja — saldo
        print("  [2/4] Sprawdzanie salda...")
        balance = await exchange.fetch_balance()
        usdt = balance.get("USDT", {})
        total = float(usdt.get("total", 0) or 0)
        free  = float(usdt.get("free",  0) or 0)
        print(f"  ✅ Saldo USDT: ${total:.2f} (dostępne: ${free:.2f})")

        # Test 3: Ticker BTC
        print("  [3/4] Pobieranie tickera BTC/USDT:USDT...")
        ticker = await exchange.fetch_ticker("BTC/USDT:USDT")
        price  = float(ticker.get("last", 0))
        print(f"  ✅ BTC cena: ${price:,.2f}")

        # Test 4: Funding rate
        print("  [4/4] Funding rate BTC...")
        try:
            fr = await exchange.fetch_funding_rate("BTC/USDT:USDT")
            rate = float(fr.get("fundingRate", 0))
            print(f"  ✅ Funding rate: {rate*100:.4f}%")
        except Exception:
            print("  ⚠️  Funding rate niedostępny (normalne na testnet)")

        print("\n" + "═"*50)
        print("  🎉 POŁĄCZENIE Z BITGET UDANE!")
        print(f"  💰 Kapitał startowy: ${free:.2f} USDT")
        print(f"  📊 Dostępne pary futures: {len(futures)}")
        print("═"*50)

        # Pokaż top 10 par futures wg wolumenu
        print("\n  Top 10 par USDT Futures (do BITGOT):")
        tickers = await exchange.fetch_tickers([s for s in futures[:50]])
        top10 = sorted(
            [(s, float(t.get("quoteVolume",0) or 0)) for s,t in tickers.items()],
            key=lambda x: x[1], reverse=True
        )[:10]
        for i, (sym, vol) in enumerate(top10, 1):
            print(f"  {i:2d}. {sym:<25} Vol: ${vol/1e6:.1f}M")

        print("\n✅ BITGOT gotowy do uruchomienia!")
        print("   Uruchom: python bitgot_main.py")
        await exchange.close()
        return True

    except ccxt.AuthenticationError as e:
        print(f"\n❌ Błąd autentykacji: {e}")
        print("   Sprawdź BITGET_PASS (hasło API)")
        await exchange.close()
        return False
    except ccxt.NetworkError as e:
        print(f"\n❌ Błąd sieci: {e}")
        await exchange.close()
        return False
    except Exception as e:
        print(f"\n❌ Błąd: {e}")
        try: await exchange.close()
        except: pass
        return False


def create_env_file():
    """Tworzy .env plik (lokalnie — NIE commituj do git!)."""
    env_content = f"""# BITGOT — Zmienne środowiskowe
# UWAGA: Nigdy nie commituj tego pliku do git!
# Dodaj .env do .gitignore

BITGET_KEY={BITGET_KEY}
BITGET_SECRET={BITGET_SECRET}
BITGET_PASS=TU_WPISZ_HASLO_API

# Tryb systemu
BITGOT_PAPER=false
BITGOT_BOTS=3000
BITGOT_CAPITAL=3000.0
"""
    with open(".env", "w") as f:
        f.write(env_content)
    print("✅ Plik .env utworzony (pamiętaj dodać go do .gitignore!)")


# ── BITGOT Config update ────────────────────────────────────────────────────────
BITGOT_CONFIG = {
    "api_key":    BITGET_KEY,
    "api_secret": BITGET_SECRET,
    "api_passphrase": BITGET_PASS,
    "exchange":   "bitget",
    "paper_mode": False,       # LIVE trading
    "testnet":    False,
    "n_bots":     3000,
    "start_capital": 3000.0,
    "base_position_pct": 0.0333,   # ~$1/bot przy $3000
    "markets": ["futures_usdt", "spot"],
}

if __name__ == "__main__":
    print("\n" + "═"*50)
    print("  BITGOT — WERYFIKACJA POŁĄCZENIA BITGET")
    print("═"*50 + "\n")

    if not check_env():
        print("\n💡 Ustaw BITGET_PASS w Replit Secrets i uruchom ponownie.")
        sys.exit(1)

    success = asyncio.run(test_connection())

    if success:
        print("\n🚀 Następny krok:")
        print("   python bitgot_main.py  ← uruchom pełny system BITGOT")
        print("\n⚠️  WAŻNE PRZYPOMNIENIE:")
        print("   1. Unieważnij stary klucz API w Bitget")
        print("   2. Wygeneruj nowy klucz (taki sam zakres uprawnień)")
        print("   3. Zaktualizuj Replit Secrets")
        print("   4. Screenshoty z kluczami = bezpieczeństwo zagrożone!")
