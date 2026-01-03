## 🚀 Installation

### 1. Download Image

```bash
docker pull leehwalmin/chiki:latest
```

### 2. Run Container

```bash
docker run -itd \
--name chiki_gogh \
--gpus all \
--retart unless-stopped \
-v /home/ec2-user/CHIKI_AI_GOGH \
-p 80:8000 \
leehwalmin/chiki:latest \
bash /root/CHIKI_AI_GOGH/server_runner.sh
```