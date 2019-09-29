---
layout: post
title:  "nginx로 swagger ui 서빙하기"
date:   2019-07-16 23:49:29 +0900
categories: ko
---

flask-apispec으로 swagger 스타일의 문서를 웹으로 제공하기 위한 설정이다.
기본 path는 swagger-ui인데, docs로 바꿔서 설정했다.
그러나, 문서 페이지를 렌더링하기 위한 각종 js, css, 이미지 파일들의 경로는 고정이기 때문에
nginx와 연동하면 제대로 렌더링이 안된다.
따라서, /flask-apispec 으로 들어오는 요청을 실제로 static files를 응답할 수 있는 엔드포인트와 연결시켜줘야 한다.




nginx를 설치한다.

```
sudo apt install nginx
```

사이트 설정을 편집한다.

```
sudo vi /etc/nginx/sites-available/my-service
```

```
server {
    listen       80;
    server_name  my-service.mdwalks.net;
    charset     utf-8;

    access_log /var/log/nginx/my-service.access.log;
    error_log  /var/log/nginx/my-service.error.log;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /api/docs/ {
        proxy_pass http://unix:/tmp/my-service.sock:/docs/;
    }
    location /flask-apispec {
        proxy_pass http://unix:/tmp/my-service.sock:/flask-apispec;
    }
    location /docs.json {
        proxy_pass http://unix:/tmp/my-service.sock:/docs.json;
    }

    location /api/ {
        proxy_set_header X-Forward-For $proxy_add_x_forwarded_for;
        proxy_set_header Host $http_host;
        proxy_pass http://unix:/tmp/my-service.sock;
        proxy_redirect off;


        client_max_body_size 200m;
    }

    location = /robots.txt {
        alias /home/linewalks/etc/robots.txt;
    }

    error_page   500 502 503 504  /50x.html;
    location = /50x.html {
        root   /usr/share/nginx/html;
    }
}
```


심볼릭 링크를 건다. 이로서 nginx에 이 설정이 사용될 준비가 되었다.
```
sudo ln -s /etc/nginx/sites-available/my-service /etc/nginx/sites-enabled/my-service
```

nginx의 설정을 다시 불러들인다.
```
sudo service nginx reload
```
