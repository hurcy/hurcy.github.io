---
layout: post
title:  "OpenStreetMap Data Wrangling with MongoDB"
date:   2016-04-10 23:49:29 +0900
categories: udacity en
---

With Seoul map data from [OppenStreetMap](https://www.openstreetmap.org), I used data munging techniques, such as assessing the quality of the data for validity, accuracy, completeness, consistency and uniformity, to clean the OpenStreetMap data for a part of the world that I care about. I used MongoDB and applied new data schema to the project.

Map Area: Seoul, Republic of Korea

https://www.openstreetmap.org/relation/2297418

https://mapzen.com/data/metro-extracts/#seoul-south-korea

### Overview
- <a href='#q1'> Problems Encountered in the Map
	- <a href='#q2'> Postal Codes
	- <a href='#q3'> Hospitals
- <a href='#q4'> Data Overview
- <a href='#q5'> Additional Ideas
	- <a href='#q6'> Translation guideline suggestion
	- <a href='#q7'> Additional data exploration using MongoDB queries
- <a href='#q8'> Conclusion

<a id='q1'></a>

#### Problems Encountered in the Map

After initially downloading a sample data of the Seoul area and running it against a provisional p3.py file, I noticed three main problems with the data, which I will discuss in the following order:

- Inconsistent postal codes (“135”, “151-742”)
- “Incorrect” postal codes (Seoul area zip codes all begin with “1” (previous format) and “01”~”09” (new format) however a large portion of all documented zip codes were outside this region)
- Inconsistent translation

<a id='q2'></a>

#### Postal Codes

Postal codes in South Korea were 6-digit numeric, and 5-digit numeric system is introduced in August 1, 2015. Postal code strings have several problems, forcing a decision to strip middle character in 6-digit numeric. (“XXX-XXX” -> “XXXXXX”).
After standarizing inconsistent postal codes, some “incorrect” postal codes surfaced when grouped together with this aggregator:

		db.seoul.aggregate([{'$match': {'tags.addr:postcode': {'$exists': 1}}}, {'$group': {'count': {'$sum': 1}, '_id': '$tags.addr:postcode'}}, {'$sort': {'count': -1}}])

Here are the top two results, beginning with the highest count:

		{u'_id': u'06321', u'count': 115},
		{u'_id': u'151841', u'count': 54}, …

Considering postal codes, it appears that 25% of documents aren't in Seoul. Seoul postal codes should begin with “1” (in 6-digit format) and “01”~”09” (in 5-digit format). I found postal code starting with “420” is the most frequently appeared in documents. It is postal code of adjacent city of Seoul. So, I performed another aggregation to check:

		db.seoul.aggregate([{'$match': {'tags.addr:city': {'$exists': 1}}}, {'$group': {'count': {'$sum': 1}, '_id': '$tags.addr:city'}}, {'$sort': {'count': 1}}])

The results:

		{u'_id': u'용인시 (Yongin)', u'count':276},
		{u'_id': u'Seoul ', u'count':234},
		{u'_id': u'연수구 ', u'count':183},
		{u'_id': u'부천시(Bucheon)', u'count':145},
		{u'_id': u'부천시 (Bucheon)' , u'count':145}, ...

These result showed that this dataset include surrounding cities such as “Bucheon”, “Yongin”. Especially, Bucheon's postal code starts with “420”, so, these postal codes aren't “incorrect”, but simply unexpected.

<a id='q3'></a>

### Hospitals

Most nodes have “name”, “name:en”, and “name:ko_rm” tags. “name:en” is written in English, and “name:ko_rm” is written as it is pronounced in Korean. I focused hospital nodes to check their name fields:

		db.seoul.find({"tags.amenity": "hospital"})

Here are the two documents:  

		{'_id': ObjectId('570a1a2c471c1f11deec607d'),
		...
		 'tags': {'amenity': 'hospital',
		 'name': '푸른정형외과의원 (Pureunjeonghyeongoegwa Clinic)',
		 'name:en': 'Pureunjeonghyeongoegwa Clinic', #Puren Orthopedic Clinic
		 'name:ko': '푸른정형외과의원',
		 'name:ko_rm': 'Pureunjeonghyeongoegwauiwon',
		 'ncat': '병원',
		 'source': 'http://kr.open.gugi.yahoo.com'},
		...
		},
		{'_id': ObjectId('570a1a2c471c1f11deec600b'),
		 ...
		 'tags': {'amenity': 'hospital',
		          'name': '우먼피아여성병원 (Umeonpiayeoseong Hospital)', 
		          'name:en': 'Umeonpiayeoseong Hospital', #Umeonpia Obstetrics & Gynecology Hospital
		          'name:ko': '우먼피아여성병원',
		          'name:ko_rm': 'Umeonpiayeoseongbyeongwon',
		          'ncat': '병원',
		          'source': 'http://kr.open.gugi.yahoo.com'},
		...
		}

“name:en” is translated name, but the documents have one as it is pronounced in Korean. So, it should be updated as I wrote in red. It is more consistent with other documents. There were typing error of “hospital” in amenity, such as “hopital”, and “hostpital”.

I summarized these cases in the following:

		mapping = { "hanuiwon": "Oriental Medicine",
		 "yeoseongbyeongwon" : "Obstetrics & Gynecology",
		 "yeoseonguiwon": "Obstetrics & Gynecology",
		 "yeoseong": "Obstetrics & Gynecology",
		 "soagwa": "Pediatric",
		 "singyeongoegwa" : "Neurosurgery",
		 "singyeong" : "Neurosurgery",
		 "jeonghyeongoegwa" : "Orthopedics",
		 "gajeonguihak": "Family Medicine",
		 "jaehwaluihakgwa": "Rehabilitation Medicine",
		 "tongjeunguihakgwa": "Pain Medicine",
		 "tongjeung": "Pain Medicine",
		 "dentist": "Dental",
		 "hopital": "Hospital",
		 "hostpital": "Hospital"
		 }

And, I updated “name:en” with new name using regex.

		db.seoul.update({'id':d}, {'$set': {"tags":{"name:en":data[d]}}})

The result, edited for readability:

		Gangdongmijeuyeoseong Hospital => Gangdongmijeu Obstetrics & Gynecology Hospital
		Pomijeuyeoseong Hospital => Pomijeu Obstetrics & Gynecology  Hospital
		Gyeonghuibom  Hanuiwon  Gangdongjeom => Gyeonghuibom   Oriental Medicine   Gangdongjeom Clinic
		...

<a id='q4'></a>

### Data Overview

This section contains basic statistics about the dataset and the MongoDB queries used to gather them.

-  File Sizes

	seoul.osm (278.7 MB)

- Number of documents

		> db.seoul.find().count() 

	1400804 

- Number of unique users

		> db.seoul.distinct("user").length 

	2353 

- Number of unique sources

		> db.seoul.distinct("tags.source").length 

	230 

- Top 1 sources

		> db.seoul.aggregate([{ "$match" : {"tags.source" : { "$exists": 1}} },{"$group":{"_id":"$tags.source", "count":{"$sum":1}}}, {"$sort": {"count":-1}}, {"$limit":1}]) 
		{ 
			"result" : [ 
				{ 
					"_id" : "http://kr.open.gugi.yahoo.com", 
					"count" : 16681 
				} 
			], 
			"ok" : 1 
		} 

<a id='q5'></a>

### Additional Ideas

<a id='q6'></a>

#### Translation guideline suggestion

The translated name field(“name:en”) contains many errors. I cleaned 13%(655/4983) of “name:en” of hospitals. Because map data is gathered throughout the world, the translation-related problem might be common in other languages including Korean. So, I think cleaning up the similar cases is necessary. 

Because of name fields are vary in different languages (“name:ko”, “name:fr”, “name:it”, etc.), common translation guideline should be considered beforehand, which might be to manage. If the translation guideline is set up and displayed to users, they might follow the guideline so that name fields are more consistent.

<a id='q7'></a>

#### Additional data exploration using MongoDB queries

##### Registration period of hospitals

		> db.seoul.aggregate( [{ "$match" : {"tags.amenity" : "hospital"} },{"$group":{"_id": "$tags.amenity", "registration_from": { $min: "$timestamp" }, "registration_to": { $max: "$timestamp" }}}])
		{ 
			"result" : [ 
				{ 
					"_id" : "hospital", 
					"registration_from" : "2009-07-28T08:18:46Z", 
					"registration_to" : "2016-03-31T17:21:41Z" 
				} 
			], 
			"ok" : 1 
		} 

##### How many  pediatric and dermatologic hospitals in seoul

		> db.seoul.aggregate( [{ "$match" : { "$or": [{"tags.name:en":{"$regex":"Pediatric"}},{"tags.name:en":{"$regex":"Dermatologic"}}] }}, {"$group":{"_id": "Pediatric and Dermatologic", "count":{"$sum":1}}}]) 
		{ 
			"result" : [ 
				{ 
					"_id" : "Pediatric and Dermatologic", 
					"count" : 415 
				} 
			], 
			"ok" : 1 
		} 

<a id='q8'></a>

### Conclusion

After this review of the map data it’s apparent that the Seoul area is incomplete, though I believe it has been well cleaned for the purposes of this exercise. It surprises me to notice documents are various structures according to area. Other metropolis have different document structure. And, a huge amount of Seoul data is copied from different sources. In the process of copying, some human mistakes seem to be inevitable. By auditing data, frequent mistakes are identified. So, I think it would be possible to prevent the mistakes through showing guidelines to users in advance, or using  data processor similar to p3.py. It helps to maintain more consistent data in OpenStreetMap.org.

