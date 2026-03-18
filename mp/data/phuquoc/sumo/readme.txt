GENERATE ROU
    python 'C:\Program Files (x86)\Eclipse\Sumo\tools\randomTrips.py' -c .\test.conf.xml

GENERATE DETECTOR
    python 'C:\Program Files (x86)\Eclipse\Sumo\tools\output\generateTLSE2Detectors.py' -n test.net.xml -l 50 -d 5 -o test_detector.add.xml