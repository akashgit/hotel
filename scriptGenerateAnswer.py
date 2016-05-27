# coding: utf-8

import datetime
from heapq import nlargest
from operator import itemgetter
from collections import defaultdict


def run_solution():
    print('Preparing arrays...')
    best_hotels_od_ulc = defaultdict(lambda: defaultdict(int))
    best_hotels_search_dest = defaultdict(lambda: defaultdict(int))
    best_hotels_search_dest1 = defaultdict(lambda: defaultdict(int))
    best_hotels_search_dest2 = defaultdict(lambda: defaultdict(int))
    best_hotels_search_dest3 = defaultdict(lambda: defaultdict(int))
    popular_hotel_cluster = defaultdict(int)
    total = 0

    # Calculate counts for groups
    with open("../data/hotel/train.csv", "r") as f:
        for line in f:
            line = line.strip()
            total += 1
            if total % 10000000 == 0:
                print('Read {} lines...'.format(total))

            if line == '':
                break

            arr = line.split(",")
            user_location_city = arr[5]
            user_location_region = arr[4]
            orig_destination_distance = arr[6]
            srch_destination_id = arr[16]
            hotel_cluster = arr[23]
            srch_children_cnt = arr[14]
            srch_adults_cnt = arr[13]
            user_id = arr[7]


            append_1 = 2

    		# Data leak flaw map, 100% accuracy on elements that have same keys in test set
            if user_location_city != '' and orig_destination_distance != '' and srch_destination_id != '':
                best_hotels_od_ulc[(user_location_city, orig_destination_distance,srch_destination_id)][hotel_cluster] += 1

    		#Destination id, user region and location groups
            if srch_destination_id != '' and user_location_city != '' and user_location_region != '' :
                best_hotels_search_dest[(srch_destination_id, user_location_city, user_location_region)][hotel_cluster] += 1

    		#Destination id, user region and user id groups, if its the same user probably looks for the same kinf of hotels
            if user_id != '' and srch_destination_id != '':
                best_hotels_search_dest1[(user_id,srch_destination_id)][hotel_cluster] += 1

    		#Destination id, user region and adults count, maybe if they are many adults few hotels can provide accomoddation(hostels) id groups, if its the same user probably looks for the same kinf of hotels
            if srch_destination_id != '' and  srch_adults_cnt != 1:
                best_hotels_search_dest2[(srch_destination_id,srch_adults_cnt)][hotel_cluster] += 1

    		#Most common hotel
            popular_hotel_cluster[hotel_cluster] += 1

	#Generate submisson file
    print('Generate submission...')
    now = datetime.datetime.now()
    path = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    out = open(path, "w")
    total = 0
    out.write("id,hotel_cluster\n")
	#get 5 most common hotel groups
    topclasters = nlargest(5, sorted(popular_hotel_cluster.items()), key=itemgetter(1))

    with open("../data/hotel/test.csv", "r") as f:
        for line in f:
            line = line.strip()
            total += 1

            if total % 1000000 == 0:
                print('Write {} lines...'.format(total))
            if line == '':
                break

            arr = line.split(",")
            id = arr[0]
            user_location_city = arr[6]
            orig_destination_distance = arr[7]
            user_location_region = arr[5]
            srch_destination_id = arr[17]
            srch_children_cnt = arr[14]
            srch_adults_cnt = arr[15]
            user_id = arr[8]

            out.write(str(id) + ',')
            filled = []

    		#tuples used as keys in generated dictionaries
            s1 = (user_location_city, orig_destination_distance, srch_destination_id)
            s2 = (srch_destination_id, user_location_city, user_location_region)
            s3 = (user_id,srch_destination_id)
            s4 = (srch_destination_id,srch_adults_cnt)

    		#If row in leaked data, give the group correponding to it
            if s1 in best_hotels_od_ulc:
                d = best_hotels_od_ulc[s1]
                topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])

    		#Give most common groups of market and user region analysis
            if s2 in best_hotels_search_dest:
                d = best_hotels_search_dest[s2]
                topitems = nlargest(5, d.items(), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])

    		#Give most common groups based on repeating search by same user
            if s3 in best_hotels_search_dest1:
                d = best_hotels_search_dest1[s3]
                topitems = nlargest(5, d.items(), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])

    		#Give most common groups based on amount of people going
            if s4 in best_hotels_search_dest2:
                d = best_hotels_search_dest2[s4]
                topitems = nlargest(5, d.items(), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])

    		#Give most common groups based on global most common
            for i in range(len(topclasters)):
                if topclasters[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topclasters[i][0])
                filled.append(topclasters[i][0])

            out.write("\n")
    print('Completed!')

run_solution()
