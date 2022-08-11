import pandas as pd
from PyPDF2 import PdfReader
import config

# Get S&P 500 index components
def get_sp_comps():
	sp_tickers = None
	print('Getting S&P 500 index components...')
	try:
		sp = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
		sp_tickers = sp['Symbol'].str.replace('.', '-').tolist()
		sp_tickers = sp_tickers + ['^GSPC']
		print('Retrieved {} tickers.'.format(len(sp_tickers)))
	except Exception as e:
		print('Error getting S&P 500 components:\n{}'.format(e))
	return sp_tickers

# Get Nasdaq 100 index components
def get_nasdaq_comps():
	nasdaq_tickers = None
	print('Getting Nasdaq 100 index components...')
	try:
		nasdaq = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100#Components')[4]
		nasdaq_tickers = nasdaq['Ticker'].tolist()
		nasdaq_tickers = nasdaq_tickers + ['NDX']
		print('Retrieved {} tickers.'.format(len(nasdaq_tickers)))
	except Exception as e:
		print('Error getting Nasdaq 100 components:\n{}'.format(e))
	return nasdaq_tickers

# Get Russell 2000 index components
def get_russell_comps():
	russell_tickers = None
	print('Getting Russell 2000 index components...')
	try:
		reader = PdfReader('data/ru2000_membershiplist_20220624_0.pdf')
		num_pages = len(reader.pages)
		russell_tickers = []
		ignore_list = [
		    '',
		    'Company', 
		    'Ticker',
		    'Membership list',
		    'Russell US Indexes',
		    'Russell 2000® Index ',
		    'June 24, 2022',
		]
		problem_comps = {
		    'IDENTIVINVE':('IDENTIV','INVE'),
		    'INDEPENDENCE REALTY TRUSTIRT':('INDEPENDENCE REALTY TRUST','IRT'),
		    'INDEPENDENT BANK CORP. MICHIGANIBCP':('INDEPENDENT BANK CORP. MICHIGAN','IBCP'),
		    'INDEPENDENT BANK GROUP INCIBTX':('INDEPENDENT BANK GROUP INC','IBTX'),
		    'INDEPENDENT BANK MASSINDB':('INDEPENDENT BANK MASS','INDB'),
		    'INDIE SEMICONDUCTOR INC (A)INDI':('INDIE SEMICONDUCTOR INC (A)','INDI'),
		    'INDUS REALTY TRUST INCINDT':('INDUS REALTY TRUST INC','INDT'),
		    'INDUSTRIAL LOGISTICS PROPERTIES TRUSTILPT':('INDUSTRIAL LOGISTICS PROPERTIES TRUST','ILPT'),
		    'INFINERAINFN':('INFINERA','INFN'),
		    'INFORMATION SERVICES GROUP INC.III':('INFORMATION SERVICES GROUP INC.','III'),
		    'INFRASTRUCTURE AND ENERGY ALTERNATIVESIEA':('INFRASTRUCTURE AND ENERGY ALTERNATIVES','IEA'),
		    'INGEVITYNGVT':('INGEVITY','NGVT'),
		    'INGLES MARKETS CL AIMKTA':('INGLES MARKETS CL A','IMKTA'),
		    'INHIBRXINBX':('INHIBRX','INBX'),
		    'INNOSPECIOSP':('INNOSPEC','IOSP'),
		    'INNOVAGE HOLDINGINNV':('INNOVAGE HOLDING','INNV'),
		    'INNOVATIVE INDUSTRIAL PROPERTIESIIPR':('INNOVATIVE INDUSTRIAL PROPERTIES','IIPR'),
		    'INNOVID CORP (A)CTV':('INNOVID CORP (A)','CTV'),
		    'INNOVIVAINVA':('INNOVIVA','INVA'),
		    'INOGEN INCINGN':('INOGEN INC','INGN'),
		    'INOTIV INCNOTV':('INOTIV INC','NOTV'),
		    'INOVIO PHARMACEUTICALS INCINO':('INOVIO PHARMACEUTICALS INC','INO'),
		    'INSEEGO CORPINSG':('INSEEGO CORP','INSG'),
		    'INSIGHT ENTERPRISESNSIT':('INSIGHT ENTERPRISES','NSIT'),
		    'INSMED INCINSM':('INSMED INC','INSM'),
		    'INSPERITYNSP':('INSPERITY','NSP'),
		    'INSPIRATO INCORPORATED (A)ISPO':('INSPIRATO INCORPORATED (A)','ISPO'),
		    'INSPIRE MEDICAL SYSTEMSINSP':('INSPIRE MEDICAL SYSTEMS','INSP'),
		    'INSPIRED ENTERTAINMENTINSE':('INSPIRED ENTERTAINMENT','INSE'),
		    'INSTALLED BUILDING PRODUCTSIBP':('INSTALLED BUILDING PRODUCTS','IBP'),
		    'INSTEEL INDSIIIN':('INSTEEL INDS','IIIN'),
		    'INSTIL BIOTIL':('INSTIL BIO','TIL'),
		    'INSTRUCTURE HOLDINGSINST':('INSTRUCTURE HOLDINGS','INST'),
		    'INTAPPINTA':('INTAPP','INTA'),
		    'INTEGER HOLDINGSITGR':('INTEGER HOLDINGS','ITGR'),
		}
		for page_num in range(num_pages-1):
		    page = reader.pages[page_num]
		    text = page.extract_text()
		    
		    clean_text = []
		    ignore_list.append(' ' + str(page_num+1))
		    for e in text.split('\n'):
		        if e in ignore_list:
		            continue
		        if e in problem_comps:
		            clean_text.append(problem_comps[e][0])
		            clean_text.append(problem_comps[e][1])
		            continue
		        clean_text.append(e)
		        
		    for i in range(len(clean_text)):
		        if i%2 != 0:
		            russell_tickers.append(clean_text[i].replace('.', '-'))

		russell_tickers = russell_tickers + ['^RUT']

		print('Retrieved {} tickers'.format(len(russell_tickers)))
	except Exception as e:
		print('Error getting Russell 2000 components:\n{}'.format(e))
	return russell_tickers


def main():
	sp_tickers = pd.DataFrame({'ticker': get_sp_comps(), 'index_name': 'S&P 500'})
	nasdaq_tickers = pd.DataFrame({'ticker': get_nasdaq_comps(), 'index_name': 'Nasdaq 100'})
	russell_tickers = pd.DataFrame({'ticker': get_russell_comps(), 'index_name': 'Russell 2000'})

	tickers = pd.concat([
		sp_tickers, 
		nasdaq_tickers, 
		russell_tickers,
	])

	tickers = tickers.groupby('ticker').agg(lambda x: x.tolist())
	tickers = tickers.rename(columns={'index_name':'indices'})
	tickers = tickers.reset_index()

	try:
		_ = tickers.to_csv(config.PATH_TO_INDEX_COMPONENTS)
		return tickers
		print('Saved index components at {}'.format(path))
	except Exception as e:
		print('Error saving index components:\n{}'.format(e))

if __name__ == '__main__':
	main()
