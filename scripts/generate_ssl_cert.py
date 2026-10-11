"""
Generates self-signed SSL certificates (cert.pem, key.pem) for local HTTPS mobile testing.
Includes SAN (Subject Alternative Names) for localhost, 127.0.0.1, and the current LAN IPv4 address.
"""

import datetime
import ipaddress
import socket
from pathlib import Path
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def get_lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Does not send actual data, merely connects to determine local route
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def generate_certificates(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    cert_path = output_dir / "cert.pem"
    key_path = output_dir / "key.pem"

    lan_ip = get_lan_ip()
    print(f"Generating self-signed certificate for LAN IP: {lan_ip} and localhost...")

    # Generate RSA private key
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )

    # Subject & Issuer
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "EG"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Giza"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Arabic Handwriting AI"),
        x509.NameAttribute(NameOID.COMMON_NAME, lan_ip),
    ])

    # SAN (Subject Alternative Names)
    san_entries = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
    ]
    if lan_ip != "127.0.0.1":
        try:
            san_entries.append(x509.IPAddress(ipaddress.IPv4Address(lan_ip)))
        except ValueError:
            san_entries.append(x509.DNSName(lan_ip))

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName(san_entries),
            critical=False,
        )
        .sign(key, hashes.SHA256(), default_backend())
    )

    # Write private key
    with open(key_path, "wb") as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    # Write certificate
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print(f"Certificates generated successfully:")
    print(f"  Cert: {cert_path}")
    print(f"  Key:  {key_path}")


if __name__ == "__main__":
    certs_dir = Path(__file__).resolve().parent.parent / "certs"
    generate_certificates(certs_dir)
