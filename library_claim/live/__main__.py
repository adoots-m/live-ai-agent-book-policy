"""Start the live server over HTTPS so a phone on the same Wi-Fi can use its camera.

    python -m library_claim.live            # https://<lan-ip>:4177/?k=<key>
    python -m library_claim.live --http     # http://127.0.0.1:4177 (desktop testing only)

Phone browsers only allow camera + microphone on https (or localhost), so a
self-signed certificate is generated into ./certs on first run. The phone will
show a certificate warning once; accept it. Alternatively tunnel with
``cloudflared tunnel --url http://localhost:4177`` and use --http.
"""

from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import logging
import socket
from pathlib import Path

import uvicorn

from ..config import ROOT


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_cert(ip: str) -> tuple[Path, Path]:
    d = ROOT / "certs"
    d.mkdir(exist_ok=True)
    crt, key = d / "dev.crt", d / "dev.key"
    if crt.exists() and key.exists():
        return crt, key
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "library-claim-agent.local")])
    san = x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                                       x509.IPAddress(ipaddress.ip_address(ip))])
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(days=1))
            .not_valid_after(now + dt.timedelta(days=365)).add_extension(san, critical=False).sign(k, hashes.SHA256()))
    crt.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                    serialization.NoEncryption()))
    return crt, key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=4177)
    ap.add_argument("--http", action="store_true", help="plain http on 127.0.0.1 (no phone)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from .server import TOKEN, app
    if a.http:
        print(f"\n  Open on this computer:  http://127.0.0.1:{a.port}/?k={TOKEN}\n")
        uvicorn.run(app, host="127.0.0.1", port=a.port, ws_max_size=8 * 1024 * 1024)
        return
    ip = lan_ip()
    crt, key = ensure_cert(ip)
    print(f"\n  Open on your phone (same Wi-Fi):  https://{ip}:{a.port}/?k={TOKEN}")
    print("  Accept the certificate warning once. Keep this window open.\n")
    uvicorn.run(app, host="0.0.0.0", port=a.port, ssl_certfile=str(crt), ssl_keyfile=str(key), ws_max_size=8 * 1024 * 1024)


if __name__ == "__main__":
    main()
