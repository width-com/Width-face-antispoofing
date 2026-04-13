# FLIP-Service

人脸活体检测服务。

上传一张人脸图片，通过三个预训练模型分别判断真伪，以多数投票得出最终结果，可识别照片翻拍、视频回放、3D 面具等常见攻击。

## Quick Start

```bash
# WSL / Linux
bash start.sh
```

启动后访问终端输出的 URL，即可使用 Web 界面上传图片测试。

## API

```
POST /predict
Content-Type: multipart/form-data
Field: image (file)
```

详见 [API_DOC.md](API_DOC.md)。
