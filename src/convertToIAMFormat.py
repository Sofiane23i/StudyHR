import os
import numpy as np
import cv2


class DataProvider():
	"this class creates machine-written text for a word list. TODO: change getNext() to return your samples."

	def __init__(self, wordList):
		self.wordList = wordList
		self.idx = 0

	def hasNext(self):
		"are there still samples to process?"
		return self.idx < len(self.wordList)

	def getNext(self):
		"TODO: return a sample from your data as a tuple containing the text and the image"
		img = np.ones((32, 128), np.uint8)*255
		word = self.wordList[self.idx]
		self.idx += 1
		cv2.putText(img, word, (2,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0), 1, cv2.LINE_AA)
		return (word, img)


def createIAMCompatibleDataset(dataProvider):
	"this function converts the passed dataset to an IAM compatible dataset"

	# create files and directories
	f = open('words.txt', 'w+')
	if not os.path.exists('sub'):
		os.makedirs('sub')
	if not os.path.exists('sub/sub-sub'):
		os.makedirs('sub/sub-sub')

	# go through data and convert it to IAM format
	ctr = 1
	while dataProvider.hasNext():
		sample = dataProvider.getNext()
		
		# write img
		cv2.imwrite('sub/sub-sub/sub-sub-%d.png'%ctr, sample[1])
		
		# write filename, dummy-values and text
		line = '%d'%ctr + ' X X X X X X X ' + sample[0] + '\n'
		f.write(line)
		
		ctr += 1
		
		
if __name__ == '__main__':
	words = ['jour',
'petithas',
'qui',
'marhe',
'jour',
'un',
'petit',
'chas',
'marche',
'le',
'chat',
'et',
'tonbe',
'dant',
'les',
'zescalié',
'et',
'il',
'pleur',
'parse',
'que',
'il',
'ses',
'fe',
'tre',
'tre',
'male',
'et',
'sa',
'maman',
'le',
'pretg',
'et',
'il',
'le',
'lech.',
'le',
'gro',
'cha',
'et',
'drmi',
'avec',
'ses',
'petis',
'chatans',
'et',
'lotr',
'cha',
'il',
'se',
'promen',
'et',
'le',
'cha',
'et',
'tonbe',
'sur',
'le',
'tapi.',
'le',
'cha',
'pler',
'et',
'lotr',
'cha',
'et',
'reviei',
'un',
'peti',
'chat',
'qui',
'marcher',
'et',
'aprer',
'boum',
'et',
'le',
'chat',
'fai',
'miaou',
'miaou',
'et',
'ça',
'mer',
'cenerv',
'et',
'ça',
'mer',
'la',
'sover',
'et',
'ail',
'rantr',
'chés',
'elle',
'avec',
'ce',
'petit',
'chat.sé',
'listoire',
'din',
'petits',
'chat.',
'qui',
'tonbe',
'partère',
'et',
'qui',
'fet',
'miaou',
'et',
'encore',
'miaou',
'puit',
'sa',
'maman',
'pran',
'son',
'petit',
'pour',
'le',
'raporté',
'son',
'petit',
'prer',
'delle',
'il',
'abide',
'dans',
'une',
'maison',
'il',
'ave',
'des',
'marche',
'le',
'petit',
'chat',
'il',
'dessandi',
'les',
'marche',
'avec',
'la',
'maman',
'chat',
'le',
'landein',
'le',
'petit',
'chat',
'désan',
'les',
'marche',
'et',
'tonbe',
'dans',
'les',
'marche',
'boum',
'sa',
'maman',
'a',
'fus',
'la',
'petit',
'chat',
'donbé',
'les',
'marche',
'et',
'le',
'petit',
'chat',
'dit',
'miaou',
'miaou',
'et',
'sa',
'maman',
'fé',
'petit',
'chat.',
'il',
'été',
'une',
'fois',
'un',
'petit',
'chat',
'qui',
'marcher.',
'il',
'avec',
'aussi',
'un',
'autre',
'chat',
'sur',
'l\'arbre',
'avec',
'ses',
'trois',
'petits',
'chatons.',
'et',
'l\'autre',
'chat',
'qui',
'marcher.',
'il',
'tonba',
'sur',
'les',
'auraille',
'et',
'dis',
'miaou',
'miaou',
'et',
'l\'autre',
'un',
'petit',
'chaton',
'qui',
'aiter',
'entrin',
'de',
'marcher',
'et',
'le',
'petit',
'chaton',
'aiter',
'tonber',
'dans',
'la',
'rut',
'-',
'et',
'le',
'chaton',
'à',
'pleurer',
'et',
'il',
'à',
'u',
'très',
'mal',
'le',
'cha',
'qui',
'dore',
'avec',
'ses',
'chaton',
'et',
'le',
'chaton',
'pareti',
'parsece',
'il',
'à',
'rivepa',
'son',
'fi',
'cont',
'il',
'déson',
'il',
'dson',
'il',
'se',
'fé',
'mal',
'poui',
'il',
'pleré.',
'poui',
'la',
'maman',
'va',
'le',
'chérché.',
'cisr',
'un',
'qui',
'c',
'uniqior',
'boum',
'miaou',
'q',
'un',
'le',
'chat',
'est',
'triste',
'et',
'se',
'tonba',
'partè',
'ape',
'se',
'mi',
'a',
'rié',
'et',
'ape',
'la',
'maman',
'chat',
'pran',
'le',
'chat.',
'le',
'chat',
'eon',
'ilavan',
'le',
'chateon',
'et',
'blécé',
'le',
'chateon',
'ipler',
'le',
'hat',
'itrin',
'un',
'chat',
'march',
'et',
'le',
'chat',
'tombe.',
'un',
'petit',
'cha',
'qui',
'desandé',
'lés',
'éscalié',
'e',
'qui',
'tonb',
'e',
'il',
'avé',
'mal',
'à',
'la',
'tet',
'e',
'sa',
'maman',
'le',
'recuper',
'avec',
'se',
'senfan.',
'le',
'cha',
'il',
'marche',
'il',
'et',
'boum',
'le',
'chas',
'il',
'coure',
'et',
'il',
'tonba',
'et',
'se',
'fais',
'male.',
'sa',
'maman',
'li',
'di',
'perpa',
'petit',
'chaton',
'et',
'elle',
'la',
'trape.',
'le',
'chat',
'marche',
'met',
'il',
'fé',
'pa',
'de',
'briu',
'boum',
'le',
'chat',
'ton.',
'et',
'la',
'maman',
'cha',
'se',
'le',
'chat',
'descend',
'les',
'escalier',
'trenquillement',
'mes',
'il',
'sur',
'tombe',
'la',
'tête',
'il',
'crie',
'jai',
'très',
'mal',
'et',
'sa',
'maman',
'le',
'rassur',
'est',
'il',
'se',
'sens',
'mieu',
'le',
'chat',
'désan',
'les',
'éscalié',
'il',
'tonb',
'sa',
'maman',
'se',
'révéle',
'et',
'la',
'maman',
'lui',
'fai',
'un',
'calin.',
'il',
'étai',
'une',
'fois',
'un',
'chat',
'ci',
'coure',
'et',
'apre',
'il',
'donbé',
'et',
'apre',
'il',
'miaou',
'miaou',
'et',
'apre',
'la',
'maman',
'la',
'atrapé.',
'et',
'le',
'chat',
'iltonbe',
'et',
'le',
'chat',
'pler',
'et',
'le',
'chat',
'se',
'fé',
'a',
'tarper',
'par',
'lotre',
'chat.',
'le',
'petit',
'chat',
'voulai',
'fair',
'son',
'malin',
'ans',
'désansan',
'les',
'marche',
'mais',
'boum',
'il',
'et',
'tonbai',
'et',
'il',
'a',
'eu',
'trai',
'mal.',
'sa',
'mamant',
'il',
'dormer',
'le',
'petit',
'chat',
'il',
'a',
'marcher',
'le',
'sele',
'chaton',
'reille',
'voulé',
'a',
'let',
'deor',
'me',
'din',
'cou',
'il',
'tonb',
'din',
'escalie',
'il',
'se',
'reléve',
'me',
'il',
'pler',
'et',
'la',
'maman',
'chat',
'se',
'recéille',
'pui',
'elle',
'le',
'reméta',
'ché',
'les',
'zotr',
'chaton.',
'un',
'peti',
'cha',
'il',
'tonbon',
'en',
'il',
'per',
'an',
'un',
'ron',
'cha',
'le',
'manje.',
'il',
'daisen',
'les',
'èscailiai',
'et',
'il',
'tonbe',
'dans',
'les',
'èscailiai',
'il',
'pleur',
'la',
'maman',
'et',
'contante',
'il',
'sa',
'muze',
'dans',
'les',
'èscaliai',
'il',
'c\'est',
'faimal',
'et',
'il',
'p',
'ere',
'et',
'il',
'et',
'eureu',
'rrrr',
'le',
'chat',
'regar',
'le',
'chat',
'la',
'maman',
'sur',
'caité',
'a',
'prai',
'elle',
'lui',
'fé',
'un',
'calun',
'il',
'ètait',
'une',
'fois',
'un',
'chat',
'qui',
'sê',
'fet.',
'male',
'et',
'boum',
'et',
'un',
'jour',
'il',
'coissa',
'miaou',
'miaou',
'et',
'sa',
'maman',
'retrouva',
'son',
'fils',
'et',
'set',
'bébé.il',
'étetu',
'fo',
'un',
'chia',
'l',
'u',
'nevou',
'les',
'passee',
'ton',
'la',
'ie',
'de',
'somere',
'éitomba',
'ilpeler',
'esomere',
'latrap',
'il',
'était',
'une',
'fois',
'un',
'petit',
'chaton',
'qui',
'kitai',
'sa',
'maman',
'il',
'marchai',
'sur',
'les',
'ais-caliai',
'il',
'tonba',
'et',
'di',
'miaou',
'miaou',
'et',
'la',
'maman',
'chate',
'elle',
'se',
'laive',
'elle',
'par',
'chairchai',
'sont',
'chaton',
'elle',
'remonte',
'ver',
'sai',
'chaton.',
'il',
'était',
'une',
'fois',
'un',
'chat',
'qui',
'marcheis',
'dran',
'quil',
'mans',
'et',
'boum',
'u',
'cé',
'fé',
'mal',
'il',
'pleurer',
'et',
'y',
'a',
'vais',
'un',
'énorme',
'chat',
'qui',
'mrde',
'le',
'chat',
'et',
'les',
'seutre',
'chats',
'il',
'regarder',
'le',
'chat.',
'un',
'peti',
'cha',
'il',
'tonbon',
'en',
'il',
'per',
'an',
'un',
'ron',
'cha',
'le',
'manje.',
'le',
'chat',
'marche',
'o',
'aiscalier',
'est',
'tonbe',
'partaire',
'le',
'chat',
'plorre',
'est',
'la',
'maman',
'requpaire',
'le',
'chat',
'arite',
'de',
'plerer',
'la',
'maman',
'chat',
'ramaine',
'le',
'chat',
'avaic',
'les',
'trois',
'aveic',
'ces',
'chat.',
'il',
'était',
'une',
'fois',
'un',
'petit',
'chat',
'il',
'marcher',
'un',
'moment',
'il',
'tanbai',
'd\'un',
'escalier',
'il',
'pleurer',
'sa',
'maman',
'le',
'raicuper',
'et',
'les',
'petit',
'chaton',
'regarder',
'et',
'la',
'maman',
'remonte',
'les',
'escalier',
'et',
'la',
'maman',
'et',
'contente',
'et',
'le',
'petit',
'chaton',
'devois',
'faire',
'attention',
'les',
'petit',
'chaton',
'sont',
'pas',
'content.'
]
	dataProvider = DataProvider(words)
	createIAMCompatibleDataset(dataProvider)
