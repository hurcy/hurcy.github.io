---
title: Ubuntu의 i3 window manager 사용 팁
layout: post
categories: ko
source-id: 1Fg3URwfF1Ewk63U8elUSpGV0MQC9AK-jveWhBPhzKEs
published: true
---
## i3 에서 desktop 없이 nautilus 실행하기 ([원문](http://askubuntu.com/questions/237953/how-to-open-nautilus-without-a-desktop))

Run this and Nautilus will always start without drawing the icons on the background.

`gsettings set org.gnome.desktop.background show-desktop-icons false`
 OR
`apt install dconf-editor`

## VirtualBox 게스트 전체화면 ([원문](https://www.virtualbox.org/ticket/14323#comment:6))

In the guest settings --> user interface --> Mini ToolBar checkboxes: I disabled the checkbox for 'Show in Full-screen/Seamless' and enabled the 'Show at Top of Screen'. After this I could resize, float the window with i3 commands and use Host + F to switch to fullscreen.

## 파일 탐색기

Ranger가 생각보다 좋다.

## 참고 문헌

* [https://i3wm.org/docs/userguide.html](https://i3wm.org/docs/userguide.html)

* [http://askubuntu.com/questions/237953/how-to-open-nautilus-without-a-desktop](http://askubuntu.com/questions/237953/how-to-open-nautilus-without-a-desktop)

* [https://www.virtualbox.org/ticket/14323#comment:6](https://www.virtualbox.org/ticket/14323#comment:6)

