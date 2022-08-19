from ast import Pass
import pandas as pd
from bs4 import BeautifulSoup
from requests_html import HTMLSession


def get_data(url):
    phones=[]
    s= HTMLSession()
    print(url)
    r=s.get(url)
    # r.html.render(sleep=1, timeout=40)
    soup= BeautifulSoup(r.html.html, 'html.parser')
    products= soup.find_all('div', {'data-component-type': 's-search-result'})
    i=0
    for product in products:
        titre_produit = product.find('a', {'class': 'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal'}).text.strip()
        
        try:
            prix = product.find('span', {'class': 'a-offscreen'}).text
        except:
            prix = 'None'
        
        try:
            rating = product.find('span', {'class': 'a-icon-alt'}).text.strip()[:4]
        except:
            Pass
        
        link = product.find('a', {'class': 'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal'})['href']
        render=s.get('https://www.amazon.fr'+link)
        render.html.render(timeout=100)
        sp= BeautifulSoup(render.html.html, 'html.parser')
        if prix=='None':
            try:
                prix = sp.find('div', {'class': 'a-section a-spacing-none aok-align-center'}).find('span', {'class': 'a-offscreen'}).text
            except:
                prix = 'None'

        try:
            marque = sp.find('div', {'class': 'a-expander-content a-expander-extend-content'}).find('td', {'class': 'a-size-base prodDetAttrValue'}).text.strip()       
        except:
            try:
                marque = sp.find('tr', 'a-spacing-small po-brand').find('td', 'a-span9').text.strip()
            except:
                marque = 'None'
        
        
        list_div_reviews = sp.find_all('div', {'data-hook': 'review-collapsed'})
        try: 
            review_text = list_div_reviews[0].find('span').text
        except:
            review_text = ""


        list_div_title = sp.find_all('a', {'data-hook': 'review-title'})
        try: 
            review_title = list_div_title[0].find('span').text
        except:
            review_title = ""


        try: 
            review_rating = sp.find('i', {'data-hook':'review-star-rating'}).text[:3]
        except:
            review_rating=''
        
        phone = {
            'titre_produit':titre_produit,
            'prix' : prix,
            'evaluations' : rating,
            'marque' : marque,
            'avis_text' : review_title+" "+review_text,
            'reviewer_evaluation': review_rating
        }
        phones.append(phone)
        i+=1
        print('phone '+str(i)+' added')
    return phones

    
n=101
urls= ['https://www.amazon.fr/s?i=electronics&rh=n%3A218193031&fs=true&page='+str(k)+'&qid=1656783194&ref=sr_pg_'+str(k) for k in range(1,n)]


for i in range(len(urls)):
    phones = get_data(urls[i])
    df = pd.DataFrame(phones).to_csv(f'data/raws/page{i+1}.csv')

