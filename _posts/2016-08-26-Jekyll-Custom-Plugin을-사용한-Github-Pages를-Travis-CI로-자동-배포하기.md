---
title: Jekyll Custom Plugin을 사용한 Github Pages를 Travis CI로 자동 배포하기
layout: post
categories: ko
source-id: 1-sa4AkDlkYD3tmT0orjEv0XAop4vcWa7gICpU9ZNp0I
published: true
---
이 포스팅은 커스텀 플러그인을 사용한 Jekyll 정적 사이트를 자동 배포하는 방법에 대해 설명한다.

나는 [tufte-jekyll theme](https://github.com/clayh53/tufte-jekyll) 이 너무 너무 쓰고 싶었다. Github이 지원하지 않는 커스텀 플러그인을 사용하려면 로컬에서 빌드한 후에 _site만 추려서 올려야 하는데, 이 과정이 번거롭기에 자동 배포를 사용하였다.


삽질 끝에 아귀를 맞추고 나니 넘 편하다. 나는 내용에만 집중해 쓰면 되고, 커밋하면 배포는 알아서 잘 돌아간다. Travis CI 짱. 도커 짱. 도커를 이렇게 일회성 배포용 환경으로 사용할 수 있는 것은 그만큼 환경 구성이 쉽기 때문이다. 

자동 배포를 위한 과정은

1) 빌드된 정적 사이트는 master 브랜치로, 전체 데이터는 source 브랜치로 [분리](http://gumpcha.github.io/blog/github-pages-with-jekyll-custom-plugin/)

2) Github의 [Personal access tokens](https://github.com/settings/tokens)

3) [Travis CI](https://travis-ci.org) 계정과 저장소 연결

4) GITHUB_API_KEY 환경변수 만들기

5) .travis.yml 파일 만들기



## Github의 Personal access tokens

Github에서 새 토큰을 발급받을 때 범위를 알맞게 설정하고, 토큰 간수를 잘 해야 한다. 나는 공개 저장소에만 접근할 수 있도록 public_repo만 선택했다. 생성된 토큰은 잘 복사해두었다가 Travis CI에 사용하면 된다.

![image alt text]({{ site.url }}/public/4DU60aE5hXK4bdCFoFxOg_img_0.png)


## Travis CI 계정과 저장소 연결

Github계정으로 Travis CI에 로그인 한다. 자동 배포할 저장소를 아래와 같이 선택한다. 

![image alt text]({{ site.url }}/public/4DU60aE5hXK4bdCFoFxOg_img_1.png)


## GITHUB_API_KEY 환경변수 만들기

Travis CI에서 연결한 저장소마다 환경변수를 추가할 수 있는데, 아래와 같이 추가하되 Display value in build log는 꼭 꺼둔다. 보안 상의 이유로 토큰의 직접적인 노출을 피하기 위해 환경변수로 만드는 것인데, 동일한 이유로, 자동 빌드할 때나 .travis.yml 에서 어떤 명령을 실행할 때에도 토큰이 콘솔에 출력되는 일은 피하는 것이 좋다. 

![image alt text]({{ site.url }}/public/4DU60aE5hXK4bdCFoFxOg_img_2.png)


## .travis.yml 파일 만들기

저장소에 .travis.yml 파일을 만들고, 빨간색으로 표시한 부분을 수정했다. [이 문서](http://stackoverflow.com/a/33125422/6760759)를 참고하여 내 환경에 맞게 수정했다. 

### .travis.yml

install: 과 script: 영역은 주석 처리한 before_script:와 script:를 사용하여 별도의 스크립트로 빼도 무방하다.
source 브랜치를 기본으로 했기에, 이 브랜치에 새로운 포스팅이 추가될 때마다 자동 배포가 될 것이다.
빌드된 파일은 _site에 있을 것이고, 이 폴더만 가지고 master 브랜치를 새로 만든다. 그리고 GITHUB_API_KEY로 저장소에 한번에 커밋한다.

```ruby

language: ruby
rvm:
- 2.3.1

# before_script:
# - chmod +x ./script/cibuild # or do this locally and commit
# Assume bundler is being used, therefore
# the `install` step will run `bundle install` by default.
# script: ./script/cibuild

install: 
  - gem install jekyll -v 3.1.6 
  - gem install rouge -v 1.11.1
  - gem install jekyll-seo-tag -v 2.0.0
script: jekyll build

# branch whitelist, only for GitHub Pages
branches:
  only:
  - source     # source branch is the main one
  
sudo: false # route your build to the container-based infrastructure for a faster build

after_success: |
  if [ -n "$GITHUB_API_KEY" ]; then
    git config user.name <username>
    git config user.email <useremail>
    git commit -am 'Automatic Update From Travis CI'
    git checkout -b master
    git filter-branch --subdirectory-filter _site/ -f
    # Make sure to make the output quiet, or else the API token will leak!
    # This works because the API key can replace your password.
    git push -fq https://$GITHUB_API_KEY@github.com/<username>/<your_repo>.git master 
  fi
``` 


## 삽질

### 삽질1

처음엔 gems의 버전을 특정하지 않고 빌드 했는데, Travis CI가 전혀 다른 버전으로 빌드를 시도하다 실패했다. gems간 버전 호환이 잘 맞지 않았던 것이 원인으로 짐작된다. 

해결책은
* 로컬에서 사용하던 버전을 특정하여 .travis.yml에 업데이트
* .travis.yml에 명시한 버전에 맞춰서 Gemfile에 업데이트

난 루비를 잘 모르지만, /var/lib/gems/2.3.0/gems 아래서 로컬에서 사용한 gem들의 버전을 확인할 수 있었다. 

Gemfile

```ruby
source 'https://rubygems.org'

gem 'jekyll', '3.1.6'
gem 'rouge', '1.11.1'
gem 'jekyll-seo-tag', '2.0.0'
``` 

### 삽질2

참고한 [이 문서](http://stackoverflow.com/a/33125422/6760759)에는 모든 표준 출력을 버리기 위해 $2>/dev/null를 썼는데, Travis CI의 쉘이 뭔가 다른지, &>/dev/null도 먹히지 않았다. 

