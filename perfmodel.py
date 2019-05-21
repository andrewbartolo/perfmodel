#!/usr/bin/env python3

import numpy as np
import sys, os


# all in SI
PARAMS = {'clockFreq':    2.8e9,
          'IPC':          1,
          'cHitLat':      5e-9,
          'mReadLat':     60e-9,
          'mWriteLat':    60e-9,
          'sendLat':      1.5e-6,
          'sendOverlap':  0
         }



CSV_HDR = ['rank', 'instrs', 'oInstrs', 'sends', 'rds', 'rdCpts', 'rdCptPct',
           'wrs', 'wrCpts', 'wrCptPct', 'cHits', 'cMisses', 'mReads', 'mWrites',
           'fullDumps', 'partDumps']

################################################################################
################################### Models #####################################
################################################################################
class BasicModel:
    def tCompute(s, proc):
        return (proc['instrs'] / PARAMS['IPC']) / PARAMS['clockFreq']

    def tCacheAndMem(s, proc):
        return proc['cHits'] * PARAMS['cHitLat'] +   \
               proc['mReads'] * PARAMS['mReadLat'] + \
               proc['mWrites'] * PARAMS['mWriteLat']

    def tCommunication(s, proc):
        return (1 - PARAMS['sendOverlap']) * proc['sends'] * PARAMS['sendLat']

    def tOverhead(s):
        return 0.1


# Fake abstract base class
class Model:
    def __init__(s, modelType, csv):
        s.m = modelType()

        s.componentTimes = np.array(list(map(s.componentify, csv)))
        processTimes = np.sum(s.componentTimes, axis=1)
        processTimes += s.m.tOverhead() # add scalar overhead to each proc time

        s.overallTime = np.max(processTimes)
        s.maxProcessTimeIdx = np.argmax(processTimes)

    def componentify(s, proc):
        return (s.m.tCompute(proc),       # times[0]
                s.m.tCacheAndMem(proc),   # times[1]
                s.m.tCommunication(proc)) # times[2]

    def breakdown(s):
        times = s.componentTimes[s.maxProcessTimeIdx]
        return {'compute': times[0] / s.overallTime,
                'cacheAndMem': times[1] / s.overallTime,
                'communication': times[2] / s.overallTime}

    def runtime(s):
        return s.overallTime
################################################################################


def parse(filename):
    # headers not automatically present for generated csvs
    # TODO some columns should be ints (not all should be floats)
    csv = np.genfromtxt(filename, delimiter=',', names=CSV_HDR)
    csv = np.sort(csv, axis=0, order='rank')

    # warn about missing parameters
    if np.sum(csv['oInstrs'] == 0):
        print("\033[93mWarn\033[0m: no overhead instructions gathered")
    if np.sum(csv['rdCpts'] == 0) or np.sum(csv['wrCpts'] == 0):
        print("\033[93mWarn\033[0m: no LD/ST sampling percentages specified")
    if np.sum(csv['mReads'] == 0) and np.sum(csv['mWrites'] == 0):
        print("\033[93mWarn\033[0m: missing memory RD/WR counts")


    print(csv)
    print('-'*40)

    mod = Model(BasicModel, csv)
    tTime = mod.runtime()
    print("Total execution time: %f" % tTime)
    print("Breakdown:", end='')
    for k, v in mod.breakdown().items():
        # output component and the percentage it contributed
        print(" [%s: %.2f%%]" % (k, v*100), end='')
    print()



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage: %s <csv>" % sys.argv[0])
        sys.exit(1)

    parse(sys.argv[1])
