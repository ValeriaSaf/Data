import json
import codecs

id = 1

def fun(filename):
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    max_reviews_count = 6800
    with open(filename, "r", encoding="utf_8_sig") as f:
        reviewsList = json.load(f)

    temp = []
    for i in reviewsList:
        # if (len(temp)>=max_reviews_count):
        #     break
        # elif ('reviewText' in i and 'overall' in i):
        #     if (i["overall"] == 1):
        #         if (count1 >= max_reviews_count/5):
        #             continue
        #         count1 +=1
        #     elif (i["overall"] == 2):
        #         if (count2 >= max_reviews_count/5):
        #             continue
        #         count2 +=1
        #     elif (i["overall"] == 3):
        #         if (count3 >= max_reviews_count/5):
        #             continue
        #         count3 +=1
        #     elif (i["overall"] == 4):
        #         if (count4 >= max_reviews_count/5):
        #             continue
        #         count4 +=1
        #     elif (i["overall"] == 5):
        #         if (count5 >= max_reviews_count/5):
        #             continue
        #         count5 +=1

            temp.append(i)

    temp = sorted(list({v['reviewText']: v for v in temp}.values()), key=lambda x: x['reviewerID'])
    for i in temp:
        global id
        i.update({"id": id,"label":0})
        id+=1

    with open("sorttest.json", "a") as w:
        json.dump(temp, w, sort_keys=True, indent=4)


f1 = open('sorttest.json', 'w')
f1.close()
#fun("AMAZON_FASHION_5.json")
fun('data_trip_Advisor.txt')
#fun("Video_Games_5.json")

with open("sorttest.json","r") as sortedReviewsJson:
    text = sortedReviewsJson.read()
    replaced = text.replace("][", ",")

with open("sorttest.json","w") as sortedReviewsJson:
    sortedReviewsJson.write(replaced)

with open("sorttest.json","r") as sortedReviewsJson:
     sortedReviewsList = json.load(sortedReviewsJson)

truncatedReviewsList = []
for i in sortedReviewsList:
    #if (len(truncatedReviewsList)<=3800):
    truncatedReviewsList.append({"id" : i["id"],"overall":i["overall"], "reviewText": i["reviewText"],"label": i["label"]})

with open("trunc.json", "w") as w:
    json.dump(truncatedReviewsList, w, sort_keys=True, indent=4)

