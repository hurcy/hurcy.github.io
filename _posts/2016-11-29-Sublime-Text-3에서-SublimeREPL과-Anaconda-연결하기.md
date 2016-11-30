---
title: Sublime Text 3에서 SublimeREPL과 Anaconda 연결하기
layout: post
author: cinyoung.hur
categories: ko
source-id: 1vmHUMilSOHqAwMwQ-Qexs3UJNsNoekKGJe-RcPMpO50
published: true
---
Sublime Text 3와 Anaconda 파이썬을 사용하는 사람들에게 이 글이 도움이 되길 바라며 정리해봤다. Sublime Text 3에서 R이나 파이썬을 대화형으로 사용하려면 SublimeREPL이 좋은 선택일 수 있다. SublimeREPL이 Anaconda 파이썬을 사용하게 하려면 설정을 약간 변경해야 한다. 

## 기본 패키지 저장소

Sublime Text 3의 경우 .deb 파일로 설치했다면, 모든 .sublime-package 파일들(zip 파일)은 기본적으로 /opt/sublime_text/Packages 에 있다.

## 사용자 패키지 설정 

패키지마다 ${packages} (== .config/sublime-text-3/Package) 아래 패키지 이름과 동일한 이름의 폴더 안에 저장된다. 직접 수정하려면 ${packages}/{패키지 이름} 에 있는 파일을 직접 수정하면 된다. 써보진 않았지만 [PackageResourceViewer](https://github.com/skuroda/PackageResourceViewer) 가 패키지 관리에 편하다고 한다. 수정 내용은 1)에서 설명한 .sublime-package 파일을 덮어 쓰게 된다. 

## SublimeREPL 설정

SublimeREPL 패키지와 관련한 파일은 ${packages}/SublimeREPL 아래 있다. 이 설정으로 인해 기존에 .sublime-package 파일을 덮어쓰는 것을 막으려면 ${packages}/User 아래에 커스터마이즈한 파일을 저장하는 것이 바람직하다. 

### Main.sublime-menu

이 [파일](https://gist.github.com/hurcy/09daa14f89bc8f9f074cf6c0c490a631)을 ${packages}/User/SublimeREPL/config/Python 아래 Main.sublime-menu 로 저장한다. "cmd" 의 값을 연결할 python 실행 파일의 위치로 바꿔준다. 

### IPython 연동을 위한 IPython 업그레이드

혹시 sublimeREPL의 IPython 실행과 함께 에러가 주루룩 뜬다면, 이 단락을 참고하라. 내가 사용하는 env 인 tm에 IPython과 Jupyter를 업그레이드 했다. 아래 버전으로 맞추지 않으면 SublimeREPL 에서 IPython 사용이 불가능하고 한다. 이 이슈가 곧 해결되길..

>> $ pip install -U ipython==4.1.1 jupyter_console==4.1.0 jupyter

### IPython 버전에 맞는 ipy_repl.py

그럼에도 여전히 거슬리는 버그가 있다. 

```

/home/user/anaconda3/envs/tm/lib/python2.7/site-packages/IPython/config.py:13: ShimWarning: The `IPython.config` package has been deprecated. You should import from traitlets.config instead.

  "You should import from traitlets.config instead.", ShimWarning)

/home/user/anaconda3/envs/tm/lib/python2.7/site-packages/IPython/terminal/console.py:13: ShimWarning: The `IPython.terminal.console` package has been deprecated. You should import from jupyter_console instead.

  "You should import from jupyter_console instead.", ShimWarning)

Jupyter Console 4.1.0

()

In [1]: 

```

이 [파일](https://gist.githubusercontent.com/MattDMo/6cb1dfbe8a124e1ca5af/raw/ec388756afcb405a55d72ae0d24f13e8d0875873/ipy_repl.py)을 복사하여 ${packages}/**User**/SublimeREPL/config/Python/ipy_repl.py 파일을 새로 만든다. 이 코드를 이용하여 sublimeREPL이 IPython을 불러오게 하려면 Main.sublime-menu 의 IPython "cmd"를 업데이트 해야한다. ([참조](https://gist.github.com/hurcy/b27d9df092645444703ed749f3e68377))

## Sublime Text 3에서 확인

Tools> SublimeREPL > Python 메뉴 아래 Anaconda의 대화형 파이썬과 IPython 이 추가된 것을 볼 수 있다.

* 대화형 파이썬

![image alt text]({{ site.url }}/public/R2CbStNwW7b8l9HNidbTMQ_img_0.png)

* IPython

![image alt text]({{ site.url }}/public/R2CbStNwW7b8l9HNidbTMQ_img_1.png)

## 참고문헌

* [http://stackoverflow.com/questions/18709422/where-are-the-default-packages-in-sublime-text-3-on-ubuntu](http://stackoverflow.com/questions/18709422/where-are-the-default-packages-in-sublime-text-3-on-ubuntu)

* [http://stackoverflow.com/questions/20861176/how-do-i-setup-sublimerepl-with-anacondas-interpreter/20861527#20861527](http://stackoverflow.com/questions/20861176/how-do-i-setup-sublimerepl-with-anacondas-interpreter/20861527#20861527)

* [https://gist.githubusercontent.com/MattDMo/6cb1dfbe8a124e1ca5af/raw/ec388756afcb405a55d72ae0d24f13e8d0875873/ipy_repl.py](https://gist.githubusercontent.com/MattDMo/6cb1dfbe8a124e1ca5af/raw/ec388756afcb405a55d72ae0d24f13e8d0875873/ipy_repl.py)

* [https://gist.github.com/hurcy/09daa14f89bc8f9f074cf6c0c490a631](https://gist.github.com/hurcy/09daa14f89bc8f9f074cf6c0c490a631)

* [https://gist.github.com/hurcy/b27d9df092645444703ed749f3e68377](https://gist.github.com/hurcy/b27d9df092645444703ed749f3e68377)



