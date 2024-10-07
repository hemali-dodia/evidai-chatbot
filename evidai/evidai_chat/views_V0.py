import requests
import google.generativeai as genai
import os
import json
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from datetime import datetime,timezone
from . import models
import logging
import hashlib
import secrets
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
from elasticsearch import Elasticsearch

# Define the connection settings
es = Elasticsearch("http://localhost:9200")  # Increased timeout

index_name = "evid_prompts"

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

key = os.environ["GOOGLE_API_KEY"] = "AIzaSyA9YvAFvikQ8MMLuF2qBgTU09Ier7KtW1U"
genai.configure(api_key="AIzaSyA9YvAFvikQ8MMLuF2qBgTU09Ier7KtW1U")

# Load the model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# Configure the logging settings
logging.basicConfig(
    filename='application.log',         # File to save logs
    level=logging.INFO,                 # Log level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    datefmt='%Y-%m-%d %H:%M:%S'         # Date format
)


def log_message(level, message):
    """
    Logs a message with a specified level and the current date and time.
    
    Args:
        level (str): The log level ('info', 'warning', 'error', etc.).
        message (str): The message to log.
    """
    logger = logging.getLogger()

    if level.lower() == 'info':
        logger.info(message)
    elif level.lower() == 'warning':
        logger.warning(message)
    elif level.lower() == 'error':
        logger.error(message)
    elif level.lower() == 'critical':
        logger.critical(message)
    else:
        logger.debug(message)  # Default to debug level for unknown levels


def hello_world(request):
    return JsonResponse({"message":"Request received successfully","data":[],"status":True},status=200)


def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()


def update_embeddings():
    prompts = models.BasicPrompts.objects.all()
    for prm in prompts:
        embedding = get_embedding(prm.prompt).tolist()
        prm.embedding = embedding
        prm.save()

# update_embeddings()

def find_most_relevant_prompt(question):
    search_query = {
        "query": {
            "multi_match": {
                "query": question,
                "fields": ["*"]
            }
        }
    }

    response = es.search(index=index_name, body=search_query)
    selected_prompt = response['hits']['hits'][0]
    return selected_prompt
    

@csrf_exempt 
def get_gemini_response(question,prompt):
        try:
            model = genai.GenerativeModel('gemini-pro')
            response_content = model.generate_content([prompt, question])
            return response_content.text.strip()
        except Exception as e:
            log_message('critical','Failed to get answer from gemini due to - '+str(e))
            return "Sorry! I am not able to find answer for your question. \nRequest you to coordinate with our support team on - hello@evident.capital.\nThank You."
    

def get_prompt_category(question,context):
    prompt = f"""Based on user's question and context, identify what is the category of this question from below mentioned categories And PROVIDE ONLY NAME OF CATEGORY NOTHING ELSE, IF NO CATEGORY MATCHES THEN RETURN "Existing_Assets" -
                 USER's QUESTION - {question}
                 CONTEXT - {context}
                 Assets_Creation: Detailed step by step process to create assets
                 Asset_Managers:This category contains information about EVIDENT's due diligence process for asset managers, the structuring and tokenization of assets, and the various fundraising methods available on the platform, emphasizing efficiency, transparency, and investor protection.
                 Onboarding_Distributor:Step by step process for distributor onboarding process
                 Onboarding_Issuer:Step by step process for issuer onboarding process
                 Corp_Investor_Onboarding:Step by step process for Corp investor onboarding process
                 Onboarding_Investor:Step by step process for investor onboarding process
                 Licensing_Custody:This category contains information about EVIDENT's regulatory compliance, including multiple licenses such as TCSP and SFC, investor protection measures, and secure handling of user funds in segregated accounts using blockchain technology.
                 Account_Opening_Funding:This category contains information about EVIDENT's streamlined and digitalized account opening process, including automated KYC and AML checks for security and regulatory compliance, and the convenient and secure procedure for depositing funds via bank transfer.
                 Security:This category contains information about EVIDENT's hybrid approach to cyber security, combining centralized and decentralized elements, the use of advanced security protocols and Algorand blockchain, compliance with financial regulations, and additional features like transaction rollbacks and IP whitelisting to protect against potential threats.
                 Product_Technology:EVIDENT is a platform for tokenizing and investing in alternative assets using blockchain technology, with features like Commitment Campaigns to raise capital, buy and sell tokenized assets, and handle asset liquidation securely. It ensures regulatory compliance and investor protection through a hybrid setup, accepting multiple currencies with spot rate exchange for non-USD funds.
                 Platform_Differentiation: EVIDENT is a fully integrated, SFC-licensed platform in Hong Kong for alternative assets, using tokenization for efficient and cost-effective investment and management of Real-World Assets.
                 Asset_Tokenization: EVIDENT tokenizes a wide range of assets, using SPVs for tangible assets and digitalization for intangible ones, ensuring regulatory compliance through "Digital Securities."
                 Digital_Transformation: EVIDENT bridges conventional financial markets and digital technologies using web3, enhancing accessibility, transparency, and efficiency with a hybrid centralized-decentralized approach.
                 Self_Custody:This category contains information about EVIDENT's support for users to self-custody their digital units in separate wallets, while ensuring compatibility with external systems, regulatory compliance, and investor protection.
                 SPV_Plan:Detailed step by step process to create SPV plan
                 SPV_Strurcture:Detailed step by step process to create SPV structure
                 About_Company:All the details about evident platfor, company and founders, services.
                 Legal_And_Regulatory:This prompt provides a comprehensive rationale for why Evident does not require a VATP license or Type 4/7/9 licenses in Hong Kong. It emphasizes the legal and regulatory positioning of Evident's hybrid model, which combines blockchain technology with centralized record-keeping, ensuring compliance with existing securities laws and the enforceability of token holders' rights.
                 Custody:This prompt outlines where customer funds are held, how they are managed and reflected on the platform, the custody practices of EVIDENT regarding underlying shares/tokens, and the transferability and tradeability of tokens outside the platform, with conditions for off-platform asset holding.
                 Structuring:This prompt explains how legal titles and share certificates are handled within EVIDENT's SPV structure, emphasizing the streamlined process for transferring legal titles at the SPV level rather than with each investor transaction.
                 Commitment_Campaign:This prompt outlines the process and mechanics of a Commitment Campaign on the EVIDENT platform, detailing how investor funds are managed during the campaign, including the handling of commitments, escrow, and the issuance of tokens.
                 buy_and_sell_tokenized_assets:This prompt describes how investors can buy and sell tokenized assets on the EVIDENT platform, covering the process of committing to investments and trading tokens within the platform's regulatory framework.
                 asset_failure_handling:This prompt explains EVIDENT's procedures for handling situations where the acquisition of target assets fails or when an asset needs to be liquidated, including the return of funds to investors and the involvement of asset managers.
                 customer_service_dispute:This prompt details how EVIDENT handles customer service and dispute resolution, highlighting the available support channels and the process for resolving disputes, including escalation if needed.
                 """
    response = get_gemini_response(question,prompt)
    return response


def identify_asset(question):
    ques = question
    assets = ['Anthropic','Canva','Databricks','Deel','Discord','Epic Games',
              'Groq','Kraken','OpenAI','Plaid','Revolut','SHEIN']
    
    cleaned_question = re.sub(r'[^a-zA-Z0-9\s]', '', ques).lower()
    total_assets = set()
    # Iterate over the assets list to find a match
    for asset in assets:
        # Convert the asset to lowercase for comparison
        if asset.lower() in cleaned_question:
            total_assets.add(asset)   # Return the matching asset
    
    return total_assets  # Return None if no match is found


def get_ipinfo():
    IPKEY = "2981978717330fdfe4548e4320bd224dc102b1d3262bf542b4a71fc6"
    url = f"https://api.ipdata.co?api-key={IPKEY}"
    response = requests.get(url)
    res = response.json()
    response = {
        "asn": res.get('asn').get('asn'),
        "asnName": res.get('asn').get('name'),
        "city": res.get('city'),
        "countryName": res.get('country_name'),
        "ip": res.get('ip'),
        "latitude": res.get('latitude'),
        "longitude": res.get('longitude'),
        "isEu": res.get('is_eu'),
        "isAnonymous": res.get('threat').get('is_anonymous'),
        "isBogon": res.get('threat').get('is_bogon'),
        "isDatacenter": res.get('threat').get('is_datacenter'),
        "isIcloudRelay": res.get('threat').get('is_icloud_relay'),
        "isKnownAbuser": res.get('threat').get('is_known_abuser'),
        "isKnownAttacker": res.get('threat').get('is_known_attacker'),
        "isProxy": res.get('threat').get('is_proxy'),
        "isThreat": res.get('threat').get('is_threat'),
        "isTor": res.get('threat').get('is_tor')
    }

    return response


@csrf_exempt
def login(request):
    try:
        if request.method=="POST":
            data = json.loads(request.body)
            user_id = data.get('user_id')
            password = data.get('password')
            # res = data.get('location_details')
            res = get_ipinfo()
            location_api = {
                            "asn": res.get('asn').get('asn'),
                            "asnName": res.get('asn').get('name'),
                            "city": res.get('city'),
                            "countryName": res.get('country_name'),
                            "ip": res.get('ip'),
                            "latitude": res.get('latitude'),
                            "longitude": res.get('longitude'),
                            "isEu": res.get('is_eu'),
                            "isAnonymous": res.get('threat').get('is_anonymous'),
                            "isBogon": res.get('threat').get('is_bogon'),
                            "isDatacenter": res.get('threat').get('is_datacenter'),
                            "isIcloudRelay": res.get('threat').get('is_icloud_relay'),
                            "isKnownAbuser": res.get('threat').get('is_known_abuser'),
                            "isKnownAttacker": res.get('threat').get('is_known_attacker'),
                            "isProxy": res.get('threat').get('is_proxy'),
                            "isThreat": res.get('threat').get('is_threat'),
                            "isTor": res.get('threat').get('is_tor')
                            }

            login_check_api = "https://api-development.evident.capital/user/login"
            data = {
                        "email": user_id,
                        "password": password,
                        "ipInfo": location_api
                    }
            response = requests.post(login_check_api, json=data)
            token = response.get('token')
            if token:
                return JsonResponse({"message":"User authenticated", "status":True, "data":{"response":"Login successfull"}},status=200)
            else:
                return JsonResponse({"message":"User authentication Failed", "status":False, "data":{"response":"Login Failed"}},status=200)
        else:           
            log_message('error', 'Invalid method request, POST method is expected.')
            return JsonResponse({"message":"Unexpected error occured","data":{
                "response":"Invalid method request, POST method is expected."},"status":False},status=200)
    except Exception as e:
        log_message('error','Failed to authenticate user due to - '+str(e))


def two_factor_authentication(request):
    try:
        if request.method=="POST":
            data = json.loads(request.body)
            token = data.get('token')
            code = data.get('code')
            headers = {
                    'Authorization': f'Bearer {token}',
                    'Content-Type': 'application/json'
                }
            login_check_api = "https://api-development.evident.capital/user/two-factor-authentication"
            data = {"code": code}
            response = requests.post(login_check_api, headers=headers, json=data)
            code = response.get('code')
            if code:
                return JsonResponse({"message":"2 Factor authentication completed", "status":True, "data":{"response":"Login successfull"}},status=200)
            else:
                return JsonResponse({"message":"You are not logged-in", "status":False, "data":{"response":"Login Failed"}},status=200)
        else:           
            log_message('error', 'Invalid method request, POST method is expected.')
            return JsonResponse({"message":"Unexpected error occured","data":{
                "response":"Invalid method request, POST method is expected."},"status":False},status=200)
    except Exception as e:
        log_message('error','Failed to authenticate user due to - '+str(e))

@csrf_exempt
def create_token(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_id = data.get("user_id")
        random_seed = secrets.token_hex(32) 
        token = hashlib.sha256(random_seed.encode()).hexdigest()
        # Save the user_id and token to the UserChatLogin model
        user_chat_login, created = models.UserChatLogin.objects.update_or_create(
            user_id=user_id,
            defaults={'token': token},
        )
        return JsonResponse({"message":"Token generated successfully","data":[{'token': token}],"status":True},status=200)

    else:
        return JsonResponse({
            "message": "Invalid JSON format",
            "status": False,
            "data": {"response":''}
        }, status=200)


@csrf_exempt
def token_validation(token):
    try:
        validate = models.UserChatLogin.objects.get(token=token)
        if validate:
            return token
        else:
            return       
    except:
        return


@csrf_exempt
def add_to_conversations(token,chat_session_id,question, answer, prompt_id):
    try:
        # Get the current date and time in UTC
        current_datetime = datetime.now(timezone.utc)

        # Convert to ISO 8601 format
        iso_format_datetime = current_datetime.isoformat()
        new_conv = models.Conversation.objects.create(
            token=token,
            chat_session_id=chat_session_id,
            question=question,
            answer=answer,
            prompt_id=prompt_id,
            created_at=iso_format_datetime,
            updated_at=iso_format_datetime
        )
        new_conv.save()
        return new_conv.id  
    except Exception as e:
        log_message('error',f'Failed to add conversation for token={token}, chat_session_is={chat_session_id},question={question},and answer={answer} due to - '+str(e))
        return None  


# Get previous conversation as context
def get_contextual_input(conversation_history, max_length=1000):
    contextual_input = '\n'.join(set(f"User_Question: {entry['question']} \nAnswer: {entry['answer']}" for entry in conversation_history))
    return contextual_input[-max_length:]


# Get all chat session based on user
@csrf_exempt
def get_chat_session_details(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        token = data.get('token')
        try:
            chats = models.ChatSession.objects.get(token=token)
            all_titles = []
            for t in chats.title:
                all_titles.append(t)
            return JsonResponse({"message":"All chat sessions fetched successfully",
                                "data":[{"response":all_titles}],"status":True},status=200)
        except Exception as e:
            log_message('error',str(e))
            return JsonResponse({"message":"No chat sessions found",
                                "data":[{"response":[]}],"status":True},status=200)
    else:
        return JsonResponse({
            "message": "Invalid JSON format",
            "status": False,
            "data": {"response":''}
        }, status=200)


# Create Session
@csrf_exempt 
def create_chat_session(request):
    try:
        # Parse JSON data from request body
        data = json.loads(request.body)
        
        # Extract required fields from data
        token = data.get('token')

        # Get the current date and time in UTC
        current_datetime = datetime.now(timezone.utc)

        # Convert to ISO 8601 format
        iso_format_datetime = current_datetime.isoformat()
        # Create a new ChatSession instance
        new_chat_session = models.ChatSession.objects.create(
            token=token,
            title="New Conversation",
            created_at=iso_format_datetime,
            updated_at=iso_format_datetime
        )
        new_chat_session.save()
        # Return a success response
        return JsonResponse({
            'message': 'Response generated successfully',
            'status': True,
            'data': {
                'response':'Chat session created successfully',
                'id': new_chat_session.id,
                'token': new_chat_session.token,
                'title': new_chat_session.title,
                'created_at': new_chat_session.created_at,
                'updated_at': new_chat_session.updated_at,
            }
        }, status=200)

    except json.JSONDecodeError:
        log_message('error','Invalid JSON format')
        return JsonResponse({
            "message": "Invalid JSON format",
            "status": False,
            "data": {"response":''}
        }, status=200)
    

# Update title of newly created chat session
@csrf_exempt
def update_chat_title(question,chat_session_id,first):
    # if request.method=='POST':
    #     data = json.loads(request)
    #     chat_session_id = data.get("chat_session_id")
    #     question = data.get("question")
        prompt = "Based on this question generate relative title for this conversation. Title should be short, it should not exceed 50 characters."
        # Get the current date and time in UTC
        current_datetime = datetime.now(timezone.utc)

        # Convert to ISO 8601 format
        iso_format_datetime = current_datetime.isoformat()
        if not first:
            title = get_gemini_response(question,prompt)
        else:
            title = "New Conversation"
        try:
            chat_session = models.ChatSession.objects.get(id=chat_session_id)
            chat_session.title = title
            chat_session.updated_at = iso_format_datetime
            chat_session.save()
            return title#JsonResponse({"message":"Title updated succesfully","data":[{"response":title}],"status":True},status=200)
        except Exception as e:
            log_message('error',str(e))
            return #JsonResponse({"message":"Failed to save chat session details",
                                #  "data":[{"response":str(e)}],"status":False},status=200)
    # else:
    #     log_message('error','Invalid JSON format')
    #     return JsonResponse({"message":"Invalid request type, POST method is expected","data":[],"status":False},status=200)


def get_conversation_for_context(chat_session_id):
    all_convo = models.Conversation.objects.filter(chat_session_id=chat_session_id)
    convo_list = [
                {"question": convo.question,
                "answer":convo.answer,"prompt_id":convo.prompt_id}
                for convo in all_convo
            ]
    return convo_list

# Get all conversation details
@csrf_exempt 
def get_conversations(request):
    if request.method == 'POST':
        data = json.loads(request.body)
    
        chat_session_id=data.get('chat_session_id')
        try:
            all_convo = models.Conversation.objects.filter(chat_session_id=chat_session_id)
            convo_list = [
                {"id": convo.id, "chat_session_id": convo.chat_session_id, "question": convo.question,
                "answer":convo.answer, "created_at":convo.created_at,"updated_at":convo.updated_at}
                for convo in all_convo
            ]
            # Construct the JSON response
            response_data = {
                "message": "Response generated successfully",
                "status": True,
                "data": {
                    "response":convo_list
                }
            }
            return JsonResponse(response_data,status=200)
        except Exception as e:
            log_message('error',str(e))
            return JsonResponse({"message": "Unexpected error occured.",
                "status": False,
                "dcata": {
                    "response":str(e)
                }},status=200)
    else:
        log_message('error', "Failed to get conversation. Invalid method, POST method is expected")
        return JsonResponse({
            "message": "Unexpected error occured.",
            "status": False,
            "data": {'response':'Invalid method, POST method is expected'}
        }, status=200)


# Validate chat session id if it is active or not
def validate_chat_session(chat_session_id):
    try:
        chat_session_id=chat_session_id
        chat_session = models.ChatSession.objects.get(id=chat_session_id)
        return chat_session
    except Exception as e:
        log_message('error', "Failed to validate chat session due to - "+str(e))
        return 


# Main flow
@csrf_exempt
def evidAI_chat(request):
    try:
        if request.method == 'POST':
            data = json.loads(request.body)
            question = data.get('question')
            chat_session_id = data.get('chat_session_id')
            token = data.get('token')
            # Validate token
            token_valid = token_validation(token)
            if token_valid is None:
                return JsonResponse({"message":"Invalid user, please login again","data":[{"response":"Failed to validate token for user, please check token or user_id"}],"status":False},status=200)

            # chat session validation
            chat_session_validation = validate_chat_session(chat_session_id)
            if chat_session_validation is None:
                log_message('error', 'Invalid chat session, kindly create new chat session')
                return JsonResponse({"message":"Unexpected error occured","data":{
                "response":"Invalid chat session, kindly create new chat session"},"status":False},status=200)
            
            # Update title    
            title_response = None
            try:      
                check_convo_number = models.Conversation.objects.get(chat_session_id=chat_session_id)
                if len(check_convo_number)<=1:
                    title_response = update_chat_title(question,chat_session_id,False)
            except Exception as e:
                title_response = update_chat_title(question,chat_session_id,True)
            
            # print("title_response - ",title_response)
            response = None
            prompt_id = set()
            conversation_history = get_conversation_for_context(chat_session_id)
            context = get_contextual_input(conversation_history, max_length=1000)
            # 1. check question category
            category = get_prompt_category(context,question)
            # print("category - ",category)
            if category =='Existing_Assets':
                asset = identify_asset(question)
                # print("asset - ",asset)
                if len(asset)>0:
                    asset_info = models.BasicPrompts.objects.filter(asset_name__in=asset)
                    # print("Found asset info - ",asset_info)
                    # Process the QuerySet
                    cleaned_asset_name = set()
                    for record in asset_info:
                        cleaned_asset_name.add(record.prompt)
                        prompt_id.add(record.id)
                    response = get_gemini_response(question,"\n".join(list(cleaned_asset_name)))
                    for id in prompt_id:
                        add_to_conversations(token,chat_session_id,question,response,id) 
                    return JsonResponse({"message":"Response generated successfully","data":{
                                                "response":response},"status":True},status=200)
                else:
                    prompt_id = list(set(int(id["prompt_id"]) for id in conversation_history if id["prompt_id"] is not None))
                    # print("prompt_id",prompt_id)
                    prom = models.BasicPrompts.objects.filter(id__in=prompt_id)
                    # print("prom - ",prom)
                    prom_desc = set()
                    for record in prom:
                        prom_desc.add(record.prompt)
                    prom_desc = "\n".join(list(prom_desc))
                    # print("prompt description generated")
                    prompt = f"""This is the context of conversation:-{context}\n\n{prom_desc}"""
                    
                    response = get_gemini_response(question,prompt)
                    if prompt_id!=[]:
                        for id in prompt_id:
                            add_to_conversations(token,chat_session_id,question,response,id) 
                    else:
                        add_to_conversations(token,chat_session_id,question,response,0) 
                    return JsonResponse({"message":"Response generated successfully","data":{
                                                "response":response},"status":True},status=200)
            else:
                try:
                    prom = models.BasicPrompts.objects.get(prompt_category=category.replace("**",""))
                    if prom.id is None:
                        prompt = f"""This is the context of conversation:-{context}\n\nYou have to search on internet and find most suitable answer for this question By refering current context. Also specify in answer that this answer is generated based on internet search as currently we dont have our data to answer this question."""
                        response = get_gemini_response(question,prompt)
                        add_to_conversations(token,chat_session_id,question,response,None)            
                        return JsonResponse({"message":"Response generated successfully","data":{
                                                "response":response},"status":True},status=200)
                    else:
                        prompt = f"""This is the context of conversation:-{context}\n\nThis is data from which you will have to generate answer-{prom.prompt}"""
                        response = get_gemini_response(question,prompt)
                        add_to_conversations(token,chat_session_id,question,response,prom.id)            
                        return JsonResponse({"message":"Response generated successfully","data":{
                                                "response":response},"status":True},status=200)
                except Exception as e:
                    log_message('error',str(e))
                    default_response = "Sorry! I am not able to find answer for your question. \nRequest you to coordinate with our support team on - hello@evident.capital.\nThank You."
                    add_to_conversations(token,chat_session_id,question,default_response,None)            
                    return JsonResponse({"message":"Response generated successfully","data":{
                                            "response":default_response},"status":True},status=200)
        else:
            log_message('error', 'Invalid method request, POST method is expected.')
            return JsonResponse({"message":"Unexpected error occured","data":{
                "response":"Invalid method request, POST method is expected."},"status":False},status=200)
    except Exception as e:
        log_message('error', str(e))
        return JsonResponse({"message":"Unexpected error occured","data":{
                "response":str(e)},"status":False},status=200)


