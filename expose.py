from pinggy import ngrok
import time

public_url = ngrok.connect(8888)
print(f"🚀 Dashboard dostępny pod adresem: {public_url}/dashboard")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    ngrok.kill()
