from __future__ import division
from __future__ import print_function

import argparse

import cv2
import editdistance

from PIL import Image

from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess

from spellchecker import SpellChecker

spell = SpellChecker(language='fr')


class FilePaths:
    "filenames and paths to data"
    fnCharList = '../model/charList.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnTrain = '../data/'
    fnInfer = '../data/298.png'
    fnCorpus = '../data/corpus.txt'


def train(model, loader):
    "train NN"
    epoch = 0  # number of training epochs since start
    bestCharErrorRate = float('inf')  # best valdiation character error rate
    noImprovementSince = 0  # number of epochs no improvement of character error rate occured
    earlyStopping = 5  # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.trainSet()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print('Batch:', iterInfo[0], '/', iterInfo[1], 'Loss:', loss)

        # validate
        charErrorRate = validate(model, loader)

        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write(
                'Validation character error rate of saved model: %f%%' % (charErrorRate * 100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1

        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
            break


def validate(model, loader):
    "validate NN"
    print('Validate NN')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0], '/', iterInfo[1])
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gtTexts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation result
    charErrorRate = numCharErr / numCharTotal if numCharTotal != 0 else 0
    wordAccuracy = numWordOK / numWordTotal if numWordTotal != 0 else 0
    print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate * 100.0, wordAccuracy * 100.0))
    return charErrorRate


def infer(model, fnImg):
    "recognize text in image provided by file path"
    imagepil = Image.open(fnImg)

    imgrgb = cv2.imread(fnImg)
    #(thresh, blackAndWhiteImage) = cv2.threshold(imgrgb, 127, 255, cv2.THRESH_BINARY)
    img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
 
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    print('Recognized:', '"' + recognized[0] + '"')
    print('Probability:', probability[0])
    misspelled = [recognized[0]]
    misspelled = spell.unknown(misspelled)
    for word in misspelled:
        print(word, spell.correction(word))
    #print(spell.correction(recognized[0]))
    #print(recognized[0])
    if(len(spell.correction(recognized[0])) > len(recognized[0])):
        print("erreur d'orthographe avec manque de lettre")
        imagepil.show()

        imagepil = imagepil.convert('RGB')

        #imagepil.show()
        indexerrorspell = []
        print(spell.correction(recognized[0]))
        print(len(spell.correction(recognized[0])))

        print((recognized[0]))
        print(len((recognized[0])))

        print(imagepil.size[0])
        jj=0
        ind= 0
        for lettre in (recognized[0]):
            print(lettre)
            #print(recognized[0][jj])
            print("")
        
            if(lettre != spell.correction(recognized[0])[jj]):
                indexerrorspell.append(jj)
            jj = jj+1    

        nbslices = imagepil.size[0]//len((recognized[0])) 
        print(nbslices)
        print(indexerrorspell)

        pixels = imagepil.load() # create the pixel map

        for i in range(nbslices*0,nbslices*(len(recognized[0])) ):   # for every col:
            for j in range(imagepil.size[1]):    # For every row
                if(pixels[i,j][0]<100):
                    pixels[i,j] = (255,0,255)

        for indspell in indexerrorspell:
            for i in range(nbslices*indspell,nbslices*(indspell+1) ):#(imagepil.size[0]):    # for every col:
                for j in range(imagepil.size[1]):    # For every row
                    if(pixels[i,j][0]<100):
                        pixels[i,j] = (255,0,0) #pixels[i,j] = (i, j, 200) # set the colour accordingly

             

        imagepil.show()
    if(len(spell.correction(recognized[0])) < len(recognized[0])):
        print("erreur d'orthographe avec des lettres supplimentaires")
        imagepil.show()

        imagepil = imagepil.convert('RGB')

        #imagepil.show()
        indexerrorspell = []
        indexcharsupp = []
        print(spell.correction(recognized[0]))
        print(len(spell.correction(recognized[0])))

        print((recognized[0]))
        print(len((recognized[0])))

        print(imagepil.size[0])
        jj=0
        for lettre in spell.correction(recognized[0]):
            print(lettre)
            print(recognized[0][jj])
            print("")
        
            if(lettre != recognized[0][jj]):
                indexerrorspell.append(jj)
            jj = jj+1
        
        suppletter = len(recognized[0]) - len(spell.correction(recognized[0]))

        nbslices = imagepil.size[0]//len((recognized[0])) 
        print(nbslices)
        print(indexerrorspell)

        pixels = imagepil.load() # create the pixel map
        for indspell in indexerrorspell:
            for i in range(nbslices*indspell,nbslices*(indspell+1) ):#(imagepil.size[0]):    # for every col:
                for j in range(imagepil.size[1]):    # For every row
                    if(pixels[i,j][0]<100):
                        pixels[i,j] = (255,0,0) #pixels[i,j] = (i, j, 200) # set the colour accordingly

        for i in range(nbslices*(len(recognized[0]) - suppletter),nbslices*(len(recognized[0])) ):   # for every col:
            for j in range(imagepil.size[1]):    # For every row
                if(pixels[i,j][0]<100):
                    pixels[i,j] = (0,255,255)

        imagepil.show()
    if(len(spell.correction(recognized[0])) == len(recognized[0])):
        print("erreur d'orthographe")

        imagepil.show()

        imagepil = imagepil.convert('RGB')

        #imagepil.show()
        indexerrorspell = []
        print(spell.correction(recognized[0]))
        print(len(spell.correction(recognized[0])))

        print((recognized[0]))
        print(len((recognized[0])))

        print(imagepil.size[0])
        jj=0
        ind= 0
        for lettre in spell.correction(recognized[0]):
            print(lettre)
            print(recognized[0][jj])
            print("")
        
            if(lettre != recognized[0][jj]):
                indexerrorspell.append(jj)
            jj = jj+1    

        nbslices = imagepil.size[0]//len((recognized[0])) 
        print(nbslices)
        print(indexerrorspell)

        pixels = imagepil.load() # create the pixel map
        for indspell in indexerrorspell:
            for i in range(nbslices*indspell,nbslices*(indspell+1) ):#(imagepil.size[0]):    # for every col:
                for j in range(imagepil.size[1]):    # For every row
                    if(pixels[i,j][0]<100):
                        pixels[i,j] = (255,0,0) #pixels[i,j] = (i, j, 200) # set the colour accordingly
             

        imagepil.show()
    #cv2.imshow('mot d\'entrée',imgrgb)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def main():
    "main function"
    # optional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train the NN', action='store_true')
    parser.add_argument('--validate', help='validate the NN', action='store_true')
    parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
    parser.add_argument('--wordbeamsearch', help='use word beam search instead of best path decoding',
                        action='store_true')
    parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')

    args = parser.parse_args()

    decoderType = DecoderType.BestPath
    if args.beamsearch:
        decoderType = DecoderType.BeamSearch
    elif args.wordbeamsearch:
        decoderType = DecoderType.WordBeamSearch

    # train or validate on IAM dataset
    if args.train or args.validate:
        # load training data, create TF model
        loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

        # save characters of model for inference mode
        open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))

        # save words contained in dataset into file
        open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

        # execute training or validation
        if args.train:
            model = Model(loader.charList, decoderType)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, decoderType, mustRestore=True)
            validate(model, loader)

    # infer text on test image
    else:
        print(open(FilePaths.fnAccuracy).read())
        model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)
        infer(model, FilePaths.fnInfer)


if __name__ == '__main__':
    main()
