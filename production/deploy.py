"""
Quick deployment script for Docker Compose stack
"""
import subprocess
import sys
import time

def run_command(cmd, description):
    """Run shell command and handle errors"""
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr}")
        return False

def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║       DPF Soot Prediction - Production Deployment       ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Build and start services
    commands = [
        ("docker-compose build", "Building Docker images"),
        ("docker-compose up -d", "Starting services"),
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            print("\n❌ Deployment failed!")
            sys.exit(1)
    
    print("\n✅ Deployment successful!")
    print("\n📡 Services running:")
    print("   • API Server:  http://localhost:8000/docs")
    print("   • Prometheus:  http://localhost:9090")
    print("   • Grafana:     http://localhost:3000 (admin/admin)")
    print("   • Redis:       localhost:6379")
    
    print("\n📝 Useful commands:")
    print("   • View logs:    docker-compose logs -f")
    print("   • Stop:         docker-compose down")
    print("   • Restart:      docker-compose restart")

if __name__ == "__main__":
    main()
