---
layout: post
title:  "gitchangelog를 사용한 릴리즈 노트 자동생성"
date:   2019-07-16 23:49:29 +0900
categories: ko
---

https://pypi.org/project/gitchangelog/

이 툴은 regex로 git commit과 pr을 요약하여 자동생성해준다.

regex로 요약하기 때문에, 애초에 커밋과 PR 메시지 작성법이 잘 잡혀있어야 깔끔한 릴리즈 노트가 나올 수 있다.

기본 설정은 크게 세가지로 커밋 메시지의 종류를 나누는데, regex 옵션을 조절하면 바꿀 수 있다.

- New: 새로운 기능
- Change: 기존 기능의 리팩토링, 기능 변경
- Fix: 버그 수정


설정은 원본 rc 파일을 참고하자.
https://github.com/vaab/gitchangelog/blob/master/src/gitchangelog/gitchangelog.rc.reference


릴리즈를 위한 과정 메모.
git 유틸이나 쉘을 만들어 쓸까 생각도 든다. 매일하는 일이 아니다보니 늘 헷갈림.

1. 브랜치 생성
2. 버전 수정(setup.py 파일 등) 후 커밋
3. git tag로 버전 명시
4. gitchangelog >> CHANGELOG.rst
5. CHANGELOG.rst 커밋
6. PR 보내기
