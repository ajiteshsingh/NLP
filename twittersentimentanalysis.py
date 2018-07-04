from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s

#consumer key, consumer secret, access token, access secret.
ckey="hIZ5x34dkydUXlgV6kLZFXcCv"
csecret="S7dzuk24dBKBElcrmPMfYqbJEzk37ETD5bVmexsIyiA7UhHDVS"
atoken="729613744167358464-ORKGiWgRt4CPBaBNRbuIqJm8t2Th5s0"
asecret="cWLWCH6A6P60eBR5G7eSVGtV48pu35qxl5M23BidHoACZ"



class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        tweet = all_data["text"]
	sentiment_value, confidence = s.sentiment(tweet)
	print(tweet, sentiment_value, confidence)

	if confidence*100 >= 80:
		output = open("twitter-out.txt","a")
		output.write(sentiment_value)
		output.write('\n')
		output.close()

	return True
        

		

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["happy"])
