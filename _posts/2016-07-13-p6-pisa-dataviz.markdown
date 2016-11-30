---
layout: post
title:  "Socio-Economic Status and Performance of Mathematics"
date:   2016-07-13 13:00:00 +0900
categories: udacity en
---

I visualized relationship of socio-economic status and performance of mathematics from Programme for International Student Assessment(PISA) dataset. The explanatory question is: What would be the average performance if all students had the OECD-average socio-economic status?

PISA measures performance only among 15-year-olds who are enrolled in education. In 2012, PISA covers 65 countries and economies,including all 34 OECD countries and 31 partner countries and economies. PISA also measures index of economic, social and cultural status, so called ESCS. It was created on the basis of the following variables: the highest levels of occupational status of the student's parents, the highest level of education of the student's parents, family wealth, etc.

According to the PISA report, we can see socio-economic status and performance in two aspects:
- How well socio-economic status predicts performance
- How much impact of socio-economic status to performance

The strength of the relationship between performance and socio-economic status varies across countries, so it is hard to compare the performance between countries. On average across OECD countries, a more socio-economically advantaged student scores 39 points higher in mathematics – the equivalent of nearly one year of schooling – than a less-advantaged student.

In addition, The countries with strong relationship between performance and socio-economic status also tend to have big performance gap across socio-economic status. On the other hand, the countires with weak relationship between them are likely have small performance gap across socio-economic status.

To compare math performance, this visualization focuses on comparisons of different education systems based on the performance of students with similar socio-economic status.

Thus, we need to assume that all students had the same OECD-average socio-economic status, and then compare the math scores between countries. Most education systems perform similarly before and after accounting for socio-economic status. However, some ranks are changed considerably. 

If socio-economic status were taken into account, rankings of mnay countries are increased. In particular, three countries that would climb more than 10 positions in their performance rankings.
- Portugal would perform above the OECD average.
- Turkey’s performance would also improve from below average to average.
- Among the partner economies, Vietnam would markedly improve its rankings.

If socio-economic status were taken into account, rankings of mnay countries are decreased. In particular, two countries that would descend more than 10 positions in their performance rankings.
- Denmark would worsen below the OECD average.
- Iceland's performance would also worsen from above to below to average.

I selected the data from "PISA 2012 Results: Excellence Through Equity", which contains very interesting insights related to equity in education. Escpecially, I read chapter 2 and chapter 5. 

# Design

## Chart types
First of all, I used **choropleth map** to address overall participating countries and economies in the dataset. With popup of each country, I presented total and sample population of PISA. 

Second, I applied **slopegraph**, because it is a good way to visualize ranking changes when socio-economic status were taken into account. You can find the answer in the slopegraph. Ranking of observed math scores is on the left side, and ranking of adjusted math scores is on the right. 

## Visual encodings
I used colors to visualize participaing countires and economies on the map.
In the slopegraph, color and hue are used to present continents differences, and ranking up/down. 

## Layouts
I used navigation tabs. Because this is martini-glass style of visualization: starting from exploratory visualization, viewers can clearly see that some countires perform better or worse given their ESCS as the navigation tabs progress.

## Legends
I added legend in the map to show categories of participating countires. In the first slopegraph, legend is to emphasis specific continent. You can hover the legend to see it.

# Feedback and Iteration
I revised my visualization for two times. 

Here are sketches:

- First sketch: http://bl.ocks.org/hurcy/raw/cd594fe884fc2a6c0812389dbb4e4a91/
- Second sketch: http://bl.ocks.org/hurcy/raw/48152dd12cb67101f0ccbc926c2de85b/
- Thrid sketch: http://bl.ocks.org/hurcy/raw/5f3a97a48c5209beedfd443af9d8e355/

Here are details:

- used popup on the choropleth map
- separated slopegraphs into three part: overall, rank up, rank down.
- updated tooltips
- updated description in each graph
- updated color hue
- adjusted indentation
- added comments in JavaScript

## Feedbacks of first sketch
Source codes are under "iter1" folder.

### feedback1
> The biggest question is how do you account for socioeconomic status? Is what ever metric you are using concrete? Does the fact that Iceland lowers mean anything? Or is it just because they have a high socioeconomic status? If I am to believe in the metric am I to believe that Vietnam, Portugal and Turkey have good education systems buy poor access?﻿

### feedback2
> I agree with Doug's input: what does it mean to beat the socioeconomic circumstances? how did you measure performance of an education system? ...
> But here you have some more input about the visualization itself.
> You say there are only three countries that climb more than 3 positions and it is hard to find those on the graph. Instead of coloring the lines according to continents you could color the lines according to increment in positions (example: green if they climb more than 5 positions, orange if between +5 and -5, red if less than -5, etc.).
> Also, is it important to visualize the continent? Is there any message you want to send such as "continent X countries would (on average) improve the most"? If this is not essential you could just drop it. If it is, you could color the words themselves according to continents, you could give the user the possibility to visualize 1 selected continent at a time (filter by continent), etc.
Ok, hope it helps :) Other than this, very cool visualization!﻿

### feedback3
> It would be great if the ESCS could be found in the visualization. And how does the adjustment change the reliability of the results, especially for countries that are far from the average ESCS? As I understand it, ESCS is mainly influenced by parents' education. But it is much harder to improve an already high level of achievement than a low level. Has this been taken into account?
> Additionally, a visualization based on the ranking of the country obfuscates the actual change of the score, which is higher for Vietnam than for Turkey.
> Last, your choice of geographic regions is poor.﻿

### feedback4
> Very good visualization! My only comments would be:
> 1. It seems that the tooltips are positioned absolutely whereas the chart is positioned relatively. So they don't always line up nicely.
> 2. What "rank" is this? If the rank (e.g. 613 for Shanghai) is a PISA score, maybe just make that explicit (and maybe give some information on what this score is for the ignorant, like me ;)﻿

### feedback5
> Very good first glance. Interesting topic. It made me want to ask 1) What's the min-max range? 2) What's the OECD global mean/median? 3) Are the score points just aligned or scaled accordingly? 4) What does "adjusted rank" mean? Does that mean for top socio-economic status group the score is much higher (higher variance - uneven society)? 5) What are the socio-economic status index of each country? 6) How about reading or other subjects?﻿

### feedback6
> Great viz! Clear annotations, good choice of colors, fonts and sizes. Well explained.
> Two things: The annotation bubbles are somewhat misaligned to the diagram. What would you say about presenting the quantiles of the math score in a box plot, instead of just listing them in the left annotation bubble?
> Best regards and keep on with the good job!﻿

## Feedbacks of second sketch
Source codes are under "iter2" folder.

### feedback1
> I like how you've broken out the slope chart into three charts to show the overall, focus on the ones who do better, and focus on the ones who do worse. I was actually just reading about a similar tactic in this blog post: http://stats.blogoverflow.com/2011/12/andyw-says-small-multiples-are-the-most-underused-data-visualization/
> The map is a little confusing. I'm not sure what OECD means, or really what I'm looking at on the map. I can mouse over countries to get the name, but nothing else pops out. I'm not sure what you're trying to accomplish. Perhaps you could pair the map side-by-side with the slope charts so that when you mouse over a country, it highlights that country's line on the chart and shows the data that goes with it?﻿

### feedback2
> Very good visualisation. A bit unclear what the map is for. I am not sure if it is needed there at all.﻿

## Feedbacks of third sketch
Source codes are under "iter3" folder.

### feedback1
>The formatting is still inconsistent.
>Please comment functional sections of your code so that it is clear what is being done.
>Make sure that your visualization has a clear message.


# Resources 
- [PISA 2012 reports][pisa]
- [More on slopegraphs][slopegraph]
- [The effect of teacher practices and attitudes][vizex1]
- [Basic Reusable Slopegraph][slopegraphex]
- [Datamap][datamap]

[pisa]: https://www.oecd.org/pisa/keyfindings/pisa-2012-results-volume-ii.htm "PISA 2012 reports"
[slopegraph]:  http://www.storytellingwithdata.com/blog/2014/03/more-on-slopegraphs  "More on slopegraphs"
[vizex1]:    https://knopthakorn.github.io/dataviz/    "The effect of teacher practices and attitudes"
[slopegraphex]: http://bl.ocks.org/biovisualize/4348024	"Basic Reusable Slopegraph"
[datamap]:	http://datamaps.github.io/	"Datamap"