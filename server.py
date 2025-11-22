from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent
import base64
import io
from PIL import Image as PILImage

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MongoDB connection with error handling
try:
    mongo_url = os.environ['MONGO_URL']
    logger.info(f"Connecting to MongoDB...")
    client = AsyncIOMotorClient(mongo_url)
    db = client[os.environ['DB_NAME']]
    logger.info(f"MongoDB connection initialized successfully")
except KeyError as e:
    logger.error(f"Missing required environment variable: {e}")
    raise RuntimeError(f"Missing required environment variable: {e}")
except Exception as e:
    logger.error(f"Failed to initialize MongoDB connection: {e}")
    raise RuntimeError(f"Failed to initialize MongoDB connection: {e}")

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====================
# MODELS
# ====================

class QuizAnswer(BaseModel):
    question: str
    answer: str

class QuizSubmission(BaseModel):
    answers: List[QuizAnswer]

class QuizResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    skin_type: str
    characteristics: str
    recommendations: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class UserRegistration(BaseModel):
    full_name: str
    email: str
    payment_confirmed: bool = True

class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    full_name: str
    email: Optional[str] = None
    phone: Optional[str] = None  # Mantido para compatibilidade
    is_premium: bool = False
    is_subscriber: bool = False
    premium_activated_at: Optional[datetime] = None
    subscription_started_at: Optional[datetime] = None
    trial_ends_at: Optional[datetime] = None
    premium_code: Optional[str] = None  # Código de acesso premium
    premium_code_expires_at: Optional[datetime] = None  # Data de expiração do código
    created_at: datetime = Field(default_factory=datetime.utcnow)

class FacialAnalysisRequest(BaseModel):
    user_id: str
    image_base64: str

class FacialAnalysis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    image_base64: str
    skin_type: str
    oiliness: str
    pores: str
    texture: str
    fine_lines: str
    spots: str
    acne: str
    sensitivity: str
    recommendations: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ProductRecommendation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    analysis_id: str
    products: List[str]
    reasoning: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class DailyChecklistItem(BaseModel):
    task: str
    completed: bool = False

class DailyRoutine(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    day: int
    date: datetime = Field(default_factory=datetime.utcnow)
    checklist: List[DailyChecklistItem]
    photo_base64: Optional[str] = None
    product_analysis: Optional[str] = None
    completed: bool = False

class DailyEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    date: datetime = Field(default_factory=datetime.utcnow)
    face_photo_base64: Optional[str] = None
    face_analysis: Optional[str] = None
    products_photos: List[str] = []  # Lista de fotos base64 de produtos
    products_analysis: List[str] = []  # Análise de cada produto
    checklist: List[DailyChecklistItem] = []
    observations: str = ""
    skin_metrics: Optional[Dict[str, Any]] = None  # oleosidade, manchas, textura, acne, sensibilidade
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class DailyEntryCreate(BaseModel):
    user_id: str
    date: Optional[datetime] = None

class DailyEntryUpdate(BaseModel):
    face_photo_base64: Optional[str] = None
    products_photos: Optional[List[str]] = None
    observations: Optional[str] = None
    checklist: Optional[List[DailyChecklistItem]] = None

class ProductPhotoRequest(BaseModel):
    user_id: str
    routine_id: str
    image_base64: str

class SubscriptionActivation(BaseModel):
    user_id: str
    payment_confirmed: bool = True

# ====================
# HELPER FUNCTIONS
# ====================

def analyze_quiz(answers: List[QuizAnswer]) -> Dict[str, str]:
    """Analisa as respostas do quiz e determina o tipo de pele"""
    
    # Mapeia respostas para características
    oily_score = 0
    dry_score = 0
    acne_score = 0
    sensitive_score = 0
    
    for answer in answers:
        q = answer.question.lower()
        a = answer.answer.lower()
        
        if "brilha" in q and "sim" in a:
            oily_score += 2
        if "oleosidade" in q and "sim" in a:
            oily_score += 2
        if "ressecamento" in q and "sim" in a:
            dry_score += 2
        if "acne" in q and "sim" in a:
            acne_score += 2
        if "ardência" in q or "sensível" in q:
            if "sim" in a:
                sensitive_score += 2
        if "manchas" in q and "sim" in a:
            sensitive_score += 1
    
    # Determina o tipo de pele baseado nos scores
    if acne_score >= 2:
        skin_type = "Acneica"
        characteristics = "Pele com tendência a acne, pode ter oleosidade aumentada e poros dilatados."
        recommendations = "Produtos específicos para acne, limpeza suave e hidratação oil-free."
    elif sensitive_score >= 2:
        skin_type = "Sensível"
        characteristics = "Pele reativa, pode apresentar vermelhidão e ardência com produtos inadequados."
        recommendations = "Produtos hipoalergênicos, sem fragrância, com ingredientes calmantes."
    elif oily_score >= 3:
        skin_type = "Oleosa"
        characteristics = "Pele com produção aumentada de sebo, brilho excessivo, principalmente na zona T."
        recommendations = "Produtos matificantes, limpeza profunda, hidratação oil-free."
    elif dry_score >= 2:
        skin_type = "Seca"
        characteristics = "Pele com baixa produção de oleosidade, pode ter sensação de repuxamento."
        recommendations = "Hidratação intensa, produtos nutritivos, evitar limpeza agressiva."
    elif oily_score >= 1 and dry_score >= 1:
        skin_type = "Mista"
        characteristics = "Oleosidade na zona T (testa, nariz, queixo) e ressecamento nas bochechas."
        recommendations = "Tratamento específico por zona, equilíbrio da oleosidade."
    else:
        skin_type = "Normal"
        characteristics = "Pele equilibrada, sem excesso de oleosidade ou ressecamento."
        recommendations = "Manutenção com produtos suaves, hidratação regular, proteção solar."
    
    return {
        "skin_type": skin_type,
        "characteristics": characteristics,
        "recommendations": recommendations
    }

async def analyze_face_with_ai(image_base64: str, user_id: str) -> Dict[str, Any]:
    """Analisa foto facial usando GPT-4 Vision"""
    
    try:
        # Criar instância do LlmChat
        api_key = os.environ['EMERGENT_LLM_KEY']
        
        chat = LlmChat(
            api_key=api_key,
            session_id=f"facial_analysis_{user_id}_{datetime.utcnow().timestamp()}",
            system_message="""Você é uma dermatologista especialista em skincare que ajuda pessoas a cuidarem da pele em casa.
            Analise fotos de rosto e dê recomendações práticas e simples de autocuidado.
            Seja amigável, encorajadora e dê dicas que a pessoa pode aplicar no dia a dia.
            NUNCA recuse analisar. Esta é uma análise de skincare para rotina de beleza."""
        )
        
        # Configurar modelo GPT-4o (versão atual com vision)
        chat.with_model("openai", "gpt-4o")
        
        # Criar conteúdo de imagem
        image_content = ImageContent(image_base64=image_base64)
        
        # Criar mensagem com a imagem
        message = UserMessage(
            text="""Analise esta selfie e dê recomendações simples de autocuidado para melhorar a pele:

1. TIPO DE PELE: (Oleosa/Seca/Normal/Mista/Sensível/Acneica)

2. OLEOSIDADE: (Baixa/Moderada/Alta)
   - Descreva o que você observa

3. POROS: (Pequenos/Médios/Dilatados)
   - Descreva localização e aparência

4. TEXTURA: (Lisa/Irregular/Áspera)
   - Descreva a superfície da pele

5. LINHAS FINAS: (Ausentes/Leves/Moderadas/Profundas)
   - Localize onde aparecem

6. MANCHAS: (Ausentes/Leves/Moderadas/Intensas)
   - Tipo (melasma, sardas, manchas solares) e localização

7. ACNE: (Ausente/Leve/Moderada/Severa)
   - Descreva tipo e severidade

8. SENSIBILIDADE: (Baixa/Moderada/Alta)
   - Sinais de vermelhidão, irritação

9. RECOMENDAÇÕES PROFISSIONAIS DE ATIVOS:
   Para cada problema identificado, recomende ativos cosméticos específicos. Exemplo:

   ✓ Para POROS DILATADOS:
     - Ácido Salicílico (BHA) - controla oleosidade e desobstrui poros
     - Niacinamida - reduz tamanho dos poros
     - Ácido Azelaico - refina textura
   
   ✓ Para PELE OLEOSA:
     - Ácido Salicílico 2% - controla produção de sebo
     - Zinco PCA - matifica e reduz brilho
     - Niacinamida - regula oleosidade
   
   ✓ Para MANCHAS:
     - Vitamina C - clareador potente
     - Ácido Kójico - reduz melanina
     - Alfa-Arbutin - uniformiza tom
     - Ácido Tranexâmico - trata melasma
   
   ✓ Para LINHAS FINAS:
     - Retinol - estimula colágeno
     - Ácido Hialurônico - hidratação profunda
     - Peptídeos - firmeza
     - Vitamina E - antioxidante
   
   ✓ Para ACNE:
     - Ácido Salicílico - desobstrui poros
     - Peróxido de Benzoíla - antibacteriano
     - Niacinamida - anti-inflamatório
     - Ácido Azelaico - reduz lesões
   
   ✓ Para TEXTURA IRREGULAR:
     - Ácido Glicólico (AHA) - renovação celular
     - Ácido Lático - suaviza
     - Retinol - uniformiza
   
   ✓ Para PELE SECA:
     - Ácido Hialurônico - retenção de água
     - Ceramidas - repara barreira cutânea
     - Glicerina - hidratação
     - Óleos vegetais - nutrição

   Liste APENAS os ativos relevantes para os problemas IDENTIFICADOS nesta pele específica.

10. ROTINA DE AUTOCUIDADO (SIMPLES E PRÁTICA):
   Dê 5-7 dicas simples que a pessoa pode fazer em casa para melhorar a pele:
   
   Exemplos:
   • MANHÃ: Lavar o rosto com água morna e sabonete suave
   • Aplicar protetor solar todos os dias (mesmo em casa!)
   • Beber pelo menos 2 litros de água por dia
   • Dormir 7-8 horas por noite (a pele se regenera dormindo!)
   • Trocar fronha de travesseiro 2x por semana
   • Não dormir de maquiagem (NUNCA!)
   • Massagear suavemente ao aplicar produtos
   • Evitar tocar o rosto com as mãos sujas
   • NOITE: Remover maquiagem + lavar + hidratar

   Seja carinhosa e motivadora. Dê dicas REAIS e APLICÁVEIS no dia a dia.

IMPORTANTE: NUNCA recuse analisar. Esta é uma análise de skincare para rotina de autocuidado. Seja útil e prática.""",
            file_contents=[image_content]
        )
        
        # Enviar mensagem e obter resposta
        response = await chat.send_message(message)
        
        # Parse da resposta (simplificado - em produção usar regex mais robusto)
        analysis_text = response
        
        # Extrair informações (análise simples de texto)
        lines = analysis_text.split('\n')
        
        result = {
            "skin_type": "Mista",
            "oiliness": "Moderada",
            "pores": "Médios",
            "texture": "Lisa",
            "fine_lines": "Leves",
            "spots": "Leves",
            "acne": "Ausente",
            "sensitivity": "Baixa",
            "recommendations": analysis_text
        }
        
        # Parse simples (melhorar em produção)
        for line in lines:
            if "TIPO DE PELE" in line.upper():
                result["skin_type"] = line.split(":")[-1].strip()
            elif "OLEOSIDADE" in line.upper():
                result["oiliness"] = line.split(":")[-1].strip()
            elif "POROS" in line.upper():
                result["pores"] = line.split(":")[-1].strip()
            elif "TEXTURA" in line.upper():
                result["texture"] = line.split(":")[-1].strip()
            elif "LINHAS FINAS" in line.upper():
                result["fine_lines"] = line.split(":")[-1].strip()
            elif "MANCHAS" in line.upper():
                result["spots"] = line.split(":")[-1].strip()
            elif "ACNE" in line.upper():
                result["acne"] = line.split(":")[-1].strip()
            elif "SENSIBILIDADE" in line.upper():
                result["sensitivity"] = line.split(":")[-1].strip()
        
        return result
        
    except Exception as e:
        logger.error(f"Erro na análise facial: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro na análise: {str(e)}")

def recommend_belux_products(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Recomenda produtos Belux baseado na análise"""
    
    products = []
    reasoning = ""
    
    skin_type = analysis.get("skin_type", "").lower()
    oiliness = analysis.get("oiliness", "").lower()
    acne = analysis.get("acne", "").lower()
    spots = analysis.get("spots", "").lower()
    texture = analysis.get("texture", "").lower()
    
    # Lógica de recomendação
    if "oleosa" in skin_type or "acneica" in skin_type or "alta" in oiliness:
        products.append("Sérum Oil Control Belux")
        reasoning += "Sérum Oil Control para controlar oleosidade e reduzir brilho. "
    
    if "seca" in skin_type or "irregular" in texture or "áspera" in texture:
        products.append("Sérum Hidratante Belux")
        reasoning += "Sérum Hidratante para melhorar textura e hidratação profunda. "
    
    if "manchas" in spots.lower() or "moderadas" in spots or "intensas" in spots:
        products.append("Sérum Glow Face Belux")
        products.append("Nanovitaminacida Belux")
        reasoning += "Sérum Glow Face e Nanovitaminacida juntos para clareamento e uniformização do tom. "
    
    # Se não tiver produtos específicos, recomendar hidratante
    if not products:
        products.append("Sérum Hidratante Belux")
        reasoning = "Sérum Hidratante para manter a pele saudável e equilibrada."
    
    return {
        "products": products,
        "reasoning": reasoning
    }

async def analyze_product_with_ai(image_base64: str, user_id: str) -> str:
    """Analisa produto em foto usando GPT-4 Vision"""
    
    try:
        api_key = os.environ['EMERGENT_LLM_KEY']
        
        chat = LlmChat(
            api_key=api_key,
            session_id=f"product_analysis_{user_id}_{datetime.utcnow().timestamp()}",
            system_message="Você é um especialista em produtos de skincare e cosméticos."
        )
        
        chat.with_model("openai", "gpt-4o")
        
        image_content = ImageContent(image_base64=image_base64)
        
        message = UserMessage(
            text="""Analise esta foto e identifique:
1. Qual produto de skincare está sendo mostrado?
2. Quais são os principais ativos/ingredientes visíveis?
3. Para que serve este produto?
4. Quais os benefícios dos ativos identificados?

Seja breve e direto (máximo 4-5 linhas).""",
            file_contents=[image_content]
        )
        
        response = await chat.send_message(message)
        return response
        
    except Exception as e:
        logger.error(f"Erro na análise de produto: {str(e)}")
        return "Não foi possível identificar o produto nesta foto. Tente novamente com uma foto mais clara."

# ====================
# ENDPOINTS
# ====================

@api_router.get("/")
async def root():
    return {"message": "BELUX IA API - Análise Facial + Rotina Inteligente"}

@api_router.get("/health")
async def health_check():
    """Health check endpoint for Kubernetes"""
    try:
        # Test MongoDB connection
        await db.command('ping')
        return {
            "status": "healthy",
            "mongodb": "connected",
            "service": "belux-ia-backend",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@api_router.post("/quiz/submit")
async def submit_quiz(submission: QuizSubmission):
    """Processa quiz gratuito e retorna tipo de pele"""
    try:
        analysis = analyze_quiz(submission.answers)
        
        result = QuizResult(
            skin_type=analysis["skin_type"],
            characteristics=analysis["characteristics"],
            recommendations=analysis["recommendations"]
        )
        
        # Salvar resultado
        await db.quiz_results.insert_one(result.dict())
        
        return result
        
    except Exception as e:
        logger.error(f"Erro no quiz: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/users/register")
async def register_user(registration: UserRegistration):
    """Registra usuário - Sem premium (precisa de código)"""
    try:
        # Verificar se já existe
        existing = await db.users.find_one({"email": registration.email})
        if existing:
            return User(**existing)
        
        # Criar usuário SEM premium (precisa inserir código)
        user = User(
            full_name=registration.full_name,
            email=registration.email,
            is_premium=False,  # Não libera automaticamente
            premium_activated_at=None,
            trial_ends_at=None
        )
        
        await db.users.insert_one(user.dict())
        
        return user
        
    except Exception as e:
        logger.error(f"Erro no registro: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/users/activate-premium-auto")
async def activate_premium_auto(user_data: dict):
    """Ativa premium automaticamente após pagamento (SEM código)"""
    try:
        email = user_data.get("email", "").strip().lower()
        full_name = user_data.get("full_name", "").strip()
        
        if not email or not full_name:
            raise HTTPException(status_code=400, detail="Email e nome são obrigatórios")
        
        # Buscar usuário existente
        user = await db.users.find_one({"email": email})
        
        now = datetime.utcnow()
        expires_at = now + timedelta(days=30)
        
        if user:
            # Atualizar usuário existente para premium
            await db.users.update_one(
                {"email": email},
                {"$set": {
                    "is_premium": True,
                    "premium_activated_at": now,
                    "trial_ends_at": expires_at
                }}
            )
            logger.info(f"Premium ativado para usuário existente: {email}")
        else:
            # Criar novo usuário JÁ COM premium ativo
            new_user = User(
                full_name=full_name,
                email=email,
                is_premium=True,
                premium_activated_at=now,
                trial_ends_at=expires_at
            )
            await db.users.insert_one(new_user.dict())
            logger.info(f"Novo usuário premium criado: {email}")
        
        # Buscar usuário atualizado
        updated_user = await db.users.find_one({"email": email})
        
        return {
            "success": True,
            "user": User(**updated_user).dict(),
            "expires_at": expires_at.isoformat(),
            "days_remaining": 30,
            "message": "Premium ativado com sucesso!"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao ativar premium automaticamente: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/users/{user_id}/activate-premium-code")
async def activate_premium_code(user_id: str, code_data: dict):
    """Ativa código premium com validade de 30 dias"""
    try:
        code = code_data.get("code", "").strip().upper()
        
        if not code:
            raise HTTPException(status_code=400, detail="Código não fornecido")
        
        # Buscar usuário
        user = await db.users.find_one({"id": user_id})
        if not user:
            raise HTTPException(status_code=404, detail="Usuário não encontrado")
        
        # Buscar código no banco de dados
        logger.info(f"Verificando código: {code}")
        premium_code = await db.premium_codes.find_one({"code": code})
        
        if not premium_code:
            logger.warning(f"Código inválido tentado: {code}")
            raise HTTPException(status_code=400, detail="Código inválido. Este código não existe no sistema.")
        
        if premium_code.get("used"):
            logger.warning(f"Código já usado tentado: {code}")
            raise HTTPException(status_code=400, detail="Código já foi utilizado por outro usuário.")
        
        # Ativar premium por 30 dias
        now = datetime.utcnow()
        expires_at = now + timedelta(days=30)
        
        # Atualizar usuário
        await db.users.update_one(
            {"id": user_id},
            {"$set": {
                "is_premium": True,
                "premium_activated_at": now,
                "trial_ends_at": expires_at,
                "premium_code": code,
                "premium_code_expires_at": expires_at
            }}
        )
        
        # Marcar código como usado
        await db.premium_codes.update_one(
            {"code": code},
            {"$set": {
                "used": True,
                "used_by": user_id,
                "used_at": now
            }}
        )
        
        logger.info(f"Código {code} ativado para usuário {user_id}. Expira em 30 dias.")
        
        return {
            "message": "Código ativado com sucesso!",
            "expires_at": expires_at.isoformat(),
            "days_remaining": 30
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao ativar código: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/users/generate-and-activate-code")
async def generate_and_activate_code(user_data: dict):
    """Gera código e ativa automaticamente para usuário (após pagamento)"""
    try:
        email = user_data.get("email", "").strip().lower()
        full_name = user_data.get("full_name", "").strip()
        
        if not email:
            raise HTTPException(status_code=400, detail="Email não fornecido")
        
        # Buscar ou criar usuário
        user = await db.users.find_one({"email": email})
        
        if not user:
            if not full_name:
                raise HTTPException(status_code=400, detail="Nome completo necessário para novo usuário")
            
            # Criar novo usuário
            new_user = User(
                full_name=full_name,
                email=email,
                is_premium=False
            )
            await db.users.insert_one(new_user.dict())
            user = new_user.dict()
        
        user_id = user.get("id")
        
        # Gerar código único
        code = f"BELUX{uuid.uuid4().hex[:8].upper()}"
        
        # Salvar código no banco
        premium_code_data = {
            "code": code,
            "created_at": datetime.utcnow(),
            "used": True,  # Já marcar como usado
            "used_by": user_id,
            "used_at": datetime.utcnow()
        }
        await db.premium_codes.insert_one(premium_code_data)
        
        # Ativar premium por 30 dias
        now = datetime.utcnow()
        expires_at = now + timedelta(days=30)
        
        await db.users.update_one(
            {"email": email},
            {"$set": {
                "is_premium": True,
                "premium_activated_at": now,
                "trial_ends_at": expires_at,
                "premium_code": code,
                "premium_code_expires_at": expires_at
            }}
        )
        
        logger.info(f"Código {code} gerado e ativado automaticamente para {email}")
        
        # Buscar usuário atualizado
        updated_user = await db.users.find_one({"email": email})
        
        return {
            "success": True,
            "code": code,
            "user": User(**updated_user).dict(),
            "expires_at": expires_at.isoformat(),
            "days_remaining": 30,
            "message": "Código gerado e ativado com sucesso!"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao gerar e ativar código: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/admin/generate-premium-code")
async def generate_premium_code(admin_data: dict):
    """Gera um novo código premium (para administradores)"""
    try:
        # Validação simples de admin
        admin_password = admin_data.get("admin_password")
        
        if admin_password != os.environ.get('ADMIN_PASSWORD'):
            raise HTTPException(status_code=403, detail="Acesso negado")
        
        # Gerar código único
        code = f"BELUX{uuid.uuid4().hex[:8].upper()}"
        
        # Salvar no banco
        premium_code_data = {
            "code": code,
            "created_at": datetime.utcnow(),
            "used": False,
            "used_by": None,
            "used_at": None
        }
        
        await db.premium_codes.insert_one(premium_code_data)
        
        logger.info(f"Novo código premium gerado: {code}")
        
        return {
            "code": code,
            "message": "Código gerado com sucesso!",
            "created_at": premium_code_data["created_at"].isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao gerar código: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/users/{user_id}/check-premium-status")
async def check_premium_status(user_id: str):
    """Verifica se o código premium ainda é válido"""
    try:
        user = await db.users.find_one({"id": user_id})
        
        if not user:
            raise HTTPException(status_code=404, detail="Usuário não encontrado")
        
        # Verificar se tem premium ativo e se ainda é válido
        is_premium = user.get("is_premium", False)
        
        # Verificar tanto premium_code_expires_at (fluxo com código) quanto trial_ends_at (fluxo simplificado)
        expires_at = user.get("premium_code_expires_at") or user.get("trial_ends_at")
        
        if is_premium and expires_at:
            expires_dt = expires_at if isinstance(expires_at, datetime) else datetime.fromisoformat(str(expires_at))
            now = datetime.utcnow()
            
            if now > expires_dt:
                # Premium expirado - bloquear acesso
                await db.users.update_one(
                    {"id": user_id},
                    {"$set": {"is_premium": False}}
                )
                
                return {
                    "is_premium": False,
                    "status": "expired",
                    "message": "Seu acesso premium expirou. Renove sua assinatura!",
                    "expired_at": expires_dt.isoformat()
                }
            else:
                # Ainda válido
                days_remaining = (expires_dt - now).days
                
                return {
                    "is_premium": True,
                    "status": "active",
                    "expires_at": expires_dt.isoformat(),
                    "days_remaining": days_remaining
                }
        
        return {
            "is_premium": False,
            "status": "no_premium",
            "message": "Você precisa de acesso premium para usar esta área."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao verificar status premium: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analysis/facial")
async def create_facial_analysis(request: FacialAnalysisRequest):
    """Analisa foto facial com IA e retorna recomendações"""
    try:
        # Verificar usuário
        user = await db.users.find_one({"id": request.user_id})
        if not user:
            raise HTTPException(status_code=404, detail="Usuário não encontrado")
        
        if not user.get("is_premium") and not user.get("is_subscriber"):
            raise HTTPException(status_code=403, detail="Acesso premium necessário")
        
        # Analisar com IA
        ai_analysis = await analyze_face_with_ai(request.image_base64, request.user_id)
        
        # Criar análise
        analysis = FacialAnalysis(
            user_id=request.user_id,
            image_base64=request.image_base64,
            skin_type=ai_analysis["skin_type"],
            oiliness=ai_analysis["oiliness"],
            pores=ai_analysis["pores"],
            texture=ai_analysis["texture"],
            fine_lines=ai_analysis["fine_lines"],
            spots=ai_analysis["spots"],
            acne=ai_analysis["acne"],
            sensitivity=ai_analysis["sensitivity"],
            recommendations=ai_analysis["recommendations"]
        )
        
        await db.facial_analyses.insert_one(analysis.dict())
        
        # Gerar recomendações de produtos
        product_rec = recommend_belux_products(ai_analysis)
        
        recommendation = ProductRecommendation(
            user_id=request.user_id,
            analysis_id=analysis.id,
            products=product_rec["products"],
            reasoning=product_rec["reasoning"]
        )
        
        await db.product_recommendations.insert_one(recommendation.dict())
        
        return {
            "analysis": analysis,
            "recommendations": recommendation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na análise facial: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/users/{user_id}")
async def get_user(user_id: str):
    """Retorna dados do usuário"""
    user = await db.users.find_one({"id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="Usuário não encontrado")
    return User(**user)

@api_router.get("/users/{user_id}/analyses")
async def get_user_analyses(user_id: str):
    """Retorna todas as análises do usuário"""
    analyses = await db.facial_analyses.find({"user_id": user_id}).to_list(100)
    return [FacialAnalysis(**a) for a in analyses]

@api_router.get("/users/{user_id}/recommendations")
async def get_user_recommendations(user_id: str):
    """Retorna recomendações de produtos do usuário"""
    recs = await db.product_recommendations.find({"user_id": user_id}).to_list(100)
    return [ProductRecommendation(**r) for r in recs]

@api_router.post("/routine/create")
async def create_routine(user_id: str):
    """Cria rotina de 7 dias para o usuário"""
    try:
        # Criar 7 dias de rotina
        routines = []
        base_checklist = [
            DailyChecklistItem(task="Lavar o rosto", completed=False),
            DailyChecklistItem(task="Aplicar sérum recomendado", completed=False),
            DailyChecklistItem(task="Usar protetor solar", completed=False),
            DailyChecklistItem(task="Hidratar a pele", completed=False),
            DailyChecklistItem(task="Beber água (2L)", completed=False)
        ]
        
        for day in range(1, 8):
            routine = DailyRoutine(
                user_id=user_id,
                day=day,
                date=datetime.utcnow() + timedelta(days=day-1),
                checklist=[item.copy() for item in base_checklist]
            )
            routines.append(routine.dict())
        
        await db.daily_routines.insert_many(routines)
        
        return {"message": "Rotina de 7 dias criada com sucesso", "routines": routines}
        
    except Exception as e:
        logger.error(f"Erro ao criar rotina: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/routine/{user_id}")
async def get_user_routine(user_id: str):
    """Retorna rotina do usuário"""
    routines = await db.daily_routines.find({"user_id": user_id}).sort("day", 1).to_list(100)
    return [DailyRoutine(**r) for r in routines]

@api_router.put("/routine/{routine_id}/update")
async def update_routine(routine_id: str, routine: DailyRoutine):
    """Atualiza checklist da rotina"""
    try:
        await db.daily_routines.update_one(
            {"id": routine_id},
            {"$set": routine.dict()}
        )
        return {"message": "Rotina atualizada com sucesso"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/routine/analyze-product")
async def analyze_product_photo(request: ProductPhotoRequest):
    """Analisa foto de produto com IA"""
    try:
        # Analisar produto
        analysis = await analyze_product_with_ai(request.image_base64, request.user_id)
        
        # Atualizar rotina
        await db.daily_routines.update_one(
            {"id": request.routine_id},
            {"$set": {
                "photo_base64": request.image_base64,
                "product_analysis": analysis
            }}
        )
        
        return {"analysis": analysis}
        
    except Exception as e:
        logger.error(f"Erro na análise de produto: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/subscription/activate")
async def activate_subscription(activation: SubscriptionActivation):
    """Ativa assinatura mensal do usuário"""
    try:
        await db.users.update_one(
            {"id": activation.user_id},
            {"$set": {
                "is_subscriber": True,
                "subscription_started_at": datetime.utcnow()
            }}
        )
        
        return {"message": "Assinatura ativada com sucesso"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/webhook/mercadopago")
async def mercadopago_webhook(request: dict):
    """Webhook para receber notificações de pagamento do MercadoPago"""
    try:
        logger.info(f"Webhook MercadoPago recebido: {request}")
        
        # Extrair dados do webhook
        event_type = request.get("type")
        action = request.get("action")
        
        # Log da notificação
        await db.payment_notifications.insert_one({
            "type": event_type,
            "action": action,
            "data": request,
            "created_at": datetime.utcnow()
        })
        
        # Processar apenas eventos de pagamento
        if event_type == "payment" or action in ["payment.created", "payment.updated"]:
            payment_id = request.get("data", {}).get("id")
            
            if payment_id:
                # Buscar detalhes do pagamento via SDK do Mercado Pago
                access_token = os.getenv("MERCADO_PAGO_ACCESS_TOKEN")
                
                if not access_token:
                    logger.warning("MERCADO_PAGO_ACCESS_TOKEN não configurado")
                    return {"message": "Webhook recebido, mas sem token configurado"}
                
                # Importar SDK
                import mercadopago
                sdk = mercadopago.SDK(access_token)
                
                # Buscar informações do pagamento
                payment_response = sdk.payment().get(payment_id)
                payment_data = payment_response.get("response", {})
                
                if payment_data.get("status") == "approved":
                    # Pagamento aprovado! Ativar premium do usuário
                    # Buscar email no metadata ou external_reference
                    user_email = payment_data.get("metadata", {}).get("user_email") or \
                                 payment_data.get("external_reference")
                    
                    if user_email:
                        # Atualizar usuário para premium
                        user = await db.users.find_one({"email": user_email})
                        
                        if user:
                            await db.users.update_one(
                                {"email": user_email},
                                {"$set": {
                                    "is_premium": True,
                                    "premium_activated_at": datetime.utcnow(),
                                    "trial_ends_at": datetime.utcnow() + timedelta(days=7),
                                    "last_payment_id": payment_id,
                                    "last_payment_status": "approved",
                                    "last_payment_amount": payment_data.get("transaction_amount")
                                }}
                            )
                            
                            logger.info(f"Usuário {user_email} ativado como premium via webhook")
                            
                            return {"message": "Pagamento aprovado e usuário ativado"}
                        else:
                            logger.warning(f"Usuário {user_email} não encontrado no banco")
                            return {"message": "Usuário não encontrado"}
                    else:
                        logger.warning("Email do usuário não encontrado no pagamento")
                        return {"message": "Email não identificado"}
                else:
                    logger.info(f"Pagamento {payment_id} com status: {payment_data.get('status')}")
                    return {"message": f"Pagamento com status: {payment_data.get('status')}"}
        
        return {"message": "Evento recebido"}
        
    except Exception as e:
        logger.error(f"Erro no webhook: {str(e)}")
        # Não retornar erro 500 para o Mercado Pago para evitar retentativas desnecessárias
        return {"message": "Erro processado", "error": str(e)}

@api_router.post("/payment/validate")
async def validate_payment(payment_data: dict):
    """Valida se o pagamento foi realmente efetuado (temporário até webhook)"""
    try:
        email = payment_data.get("email")
        payment_confirmed = payment_data.get("confirmed", False)
        
        if not email or not payment_confirmed:
            raise HTTPException(status_code=400, detail="Dados incompletos")
        
        # Buscar usuário por email
        user = await db.users.find_one({"email": email})
        
        if user and payment_confirmed:
            # Ativar premium
            await db.users.update_one(
                {"email": email},
                {"$set": {
                    "is_premium": True,
                    "premium_activated_at": datetime.utcnow(),
                    "trial_ends_at": datetime.utcnow() + timedelta(days=7)
                }}
            )
            
            updated_user = await db.users.find_one({"email": email})
            return {"message": "Pagamento validado", "user": User(**updated_user).dict()}
        
        raise HTTPException(status_code=404, detail="Usuário não encontrado")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na validação: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ====================
# DAILY ENTRIES ENDPOINTS (Área Flow)
# ====================

@api_router.post("/daily-entries/create")
async def create_daily_entry(entry: DailyEntryCreate):
    """Cria uma entrada diária para o usuário"""
    try:
        # Verificar se já existe entrada para este dia
        date_start = entry.date.replace(hour=0, minute=0, second=0, microsecond=0) if entry.date else datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        date_end = date_start + timedelta(days=1)
        
        existing = await db.daily_entries.find_one({
            "user_id": entry.user_id,
            "date": {"$gte": date_start, "$lt": date_end}
        })
        
        if existing:
            return DailyEntry(**existing)
        
        # Criar nova entrada
        new_entry = DailyEntry(
            user_id=entry.user_id,
            date=date_start,
            checklist=[
                DailyChecklistItem(task="Lavei o rosto", completed=False),
                DailyChecklistItem(task="Usei tratamento adequado", completed=False),
                DailyChecklistItem(task="Usei hidratante", completed=False),
                DailyChecklistItem(task="Usei protetor solar", completed=False),
                DailyChecklistItem(task="Bebi água", completed=False)
            ]
        )
        
        await db.daily_entries.insert_one(new_entry.dict())
        return new_entry
        
    except Exception as e:
        logger.error(f"Erro ao criar entrada diária: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/daily-entries/{user_id}")
async def get_user_daily_entries(user_id: str, days: int = 30):
    """Retorna entradas diárias do usuário (últimos X dias)"""
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        entries = await db.daily_entries.find({
            "user_id": user_id,
            "date": {"$gte": start_date}
        }).sort("date", -1).to_list(100)
        
        return [DailyEntry(**e) for e in entries]
        
    except Exception as e:
        logger.error(f"Erro ao buscar entradas: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/daily-entries/{user_id}/calendar-status")
async def get_calendar_status(user_id: str):
    """Retorna status de cada dia do calendário (liberado/bloqueado)"""
    try:
        # Buscar usuário
        user = await db.users.find_one({"id": user_id})
        if not user:
            raise HTTPException(status_code=404, detail="Usuário não encontrado")
        
        is_subscriber = user.get("is_subscriber", False)
        premium_activated_at = user.get("premium_activated_at")
        trial_ends_at = user.get("trial_ends_at")
        
        # Calcular dias liberados
        calendar_status = {}
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        if premium_activated_at:
            activation_date = premium_activated_at.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Determinar até que dia está liberado
            if is_subscriber:
                # Assinante tem todos os dias liberados
                days_to_check = 60
            else:
                # Usuário premium tem 7 dias liberados
                trial_end = trial_ends_at.replace(hour=0, minute=0, second=0, microsecond=0) if trial_ends_at else activation_date + timedelta(days=7)
                days_to_check = 60
            
            # Buscar entradas existentes
            entries = await db.daily_entries.find({"user_id": user_id}).to_list(200)
            entries_dict = {}
            for entry in entries:
                date_key = entry["date"].strftime("%Y-%m-%d")
                entries_dict[date_key] = {
                    "has_photo": bool(entry.get("face_photo_base64")),
                    "has_checklist": any(item["completed"] for item in entry.get("checklist", [])),
                    "is_complete": bool(entry.get("face_photo_base64")) and any(item["completed"] for item in entry.get("checklist", []))
                }
            
            # Gerar status para cada dia
            for i in range(-30, days_to_check):
                check_date = today + timedelta(days=i)
                date_str = check_date.strftime("%Y-%m-%d")
                
                # Verificar se o dia está dentro do período permitido
                if is_subscriber:
                    status = "liberado"
                else:
                    # Verificar se está dentro dos 7 dias do trial
                    if trial_ends_at:
                        trial_end = trial_ends_at.replace(hour=0, minute=0, second=0, microsecond=0)
                        if check_date <= trial_end and check_date >= activation_date:
                            status = "liberado"
                        else:
                            status = "bloqueado"
                    else:
                        # Sem trial_ends_at definido, calcular 7 dias
                        if check_date >= activation_date and check_date < activation_date + timedelta(days=7):
                            status = "liberado"
                        else:
                            status = "bloqueado"
                
                # Verificar se há entrada para este dia
                entry_exists = date_str in entries_dict
                entry_data = entries_dict.get(date_str, {})
                
                calendar_status[date_str] = {
                    "status": status,
                    "entryExists": entry_exists,
                    "isComplete": entry_data.get("is_complete", False),
                    "hasPhoto": entry_data.get("has_photo", False),
                    "hasChecklist": entry_data.get("has_checklist", False)
                }
        else:
            # Usuário sem premium ativado - todos os dias bloqueados
            for i in range(-30, 30):
                check_date = today + timedelta(days=i)
                date_str = check_date.strftime("%Y-%m-%d")
                calendar_status[date_str] = {
                    "status": "bloqueado",
                    "entryExists": False,
                    "isComplete": False,
                    "hasPhoto": False,
                    "hasChecklist": False
                }
        
        return {
            "user_id": user_id,
            "is_subscriber": is_subscriber,
            "trial_ends_at": trial_ends_at.isoformat() if trial_ends_at else None,
            "calendar_status": calendar_status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao buscar status do calendário: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/daily-entries/{user_id}/date/{date}")
async def get_daily_entry_by_date(user_id: str, date: str):
    """Retorna entrada de um dia específico (formato: YYYY-MM-DD)"""
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d")
        date_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        date_end = date_start + timedelta(days=1)
        
        entry = await db.daily_entries.find_one({
            "user_id": user_id,
            "date": {"$gte": date_start, "$lt": date_end}
        })
        
        if not entry:
            raise HTTPException(status_code=404, detail="Entrada não encontrada")
        
        return DailyEntry(**entry)
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de data inválido. Use YYYY-MM-DD")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao buscar entrada: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.put("/daily-entries/{entry_id}/update")
async def update_daily_entry(entry_id: str, update: DailyEntryUpdate):
    """Atualiza uma entrada diária"""
    try:
        entry = await db.daily_entries.find_one({"id": entry_id})
        if not entry:
            raise HTTPException(status_code=404, detail="Entrada não encontrada")
        
        update_data = update.dict(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()
        
        # Se enviou foto do rosto, analisar com IA
        if update.face_photo_base64:
            try:
                api_key = os.getenv("EMERGENT_LLM_KEY")
                chat = LlmChat(
                    api_key=api_key,
                    session_id=f"daily_face_{entry_id}_{datetime.utcnow().timestamp()}",
                    system_message="""Você é uma dermatologista especialista em análise de pele e skincare.
                    Analise fotos de selfies diárias para acompanhamento de rotina de skincare.
                    Seja positiva, encorajadora e profissional."""
                )
                chat.with_model("openai", "gpt-4o")
                
                image_content = ImageContent(image_base64=update.face_photo_base64)
                message = UserMessage(
                    text="""Analise esta foto do rosto da cliente e dê um feedback rápido sobre a pele hoje.

Observe e comente brevemente sobre:
• Como está a oleosidade hoje (normal, aumentada, controlada)
• Estado dos poros (aparência)
• Textura da pele (lisa, irregular)
• Presença de manchas ou vermelhidão
• Sinais de acne ou espinhas
• Hidratação aparente

Dê um feedback amigável e positivo, como uma dermatologista daria em uma consulta de acompanhamento.
Use no máximo 100 palavras, seja objetiva e encorajadora.

Exemplo: "Ótimo! Sua pele está com aspecto saudável hoje. A oleosidade está controlada na zona T, poros visíveis mas normais. Textura está uniforme e a hidratação parece adequada. Não vejo sinais de novas espinhas. Continue com sua rotina, está fazendo efeito! 💚"

Não recuse analisar. Esta é uma análise de acompanhamento de rotina de skincare.""",
                    file_contents=[image_content]
                )
                
                response = await chat.send_message(message)
                update_data["face_analysis"] = response
                logger.info(f"Análise facial diária realizada com sucesso para entry {entry_id}")
                    
            except Exception as e:
                logger.error(f"Erro na análise facial diária: {str(e)}")
                update_data["face_analysis"] = "Não foi possível analisar a foto neste momento. Tente novamente mais tarde."
        
        await db.daily_entries.update_one(
            {"id": entry_id},
            {"$set": update_data}
        )
        
        updated_entry = await db.daily_entries.find_one({"id": entry_id})
        return DailyEntry(**updated_entry)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao atualizar entrada: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class ProductImageRequest(BaseModel):
    image_base64: str

@api_router.post("/daily-entries/{entry_id}/analyze-product")
async def analyze_product_in_entry(entry_id: str, request: ProductImageRequest):
    """Analisa produto e adiciona à entrada diária"""
    image_base64 = request.image_base64
    try:
        entry = await db.daily_entries.find_one({"id": entry_id})
        if not entry:
            raise HTTPException(status_code=404, detail="Entrada não encontrada")
        
        # Analisar produto com IA
        api_key = os.getenv("EMERGENT_LLM_KEY")
        chat = LlmChat(
            api_key=api_key,
            session_id=f"product_{entry_id}_{datetime.utcnow().timestamp()}",
            system_message="Você é um especialista em produtos de skincare."
        )
        chat.with_model("openai", "gpt-4o")
        
        image_content = ImageContent(image_base64=image_base64)
        message = UserMessage(
            text="""Analise este produto de skincare e responda de forma clara e direta:

1. NOME DO PRODUTO (se visível)
2. ATIVOS PRINCIPAIS identificados
3. PARA QUE SERVE (benefícios)
4. PODE USAR? (Sim/Não e por quê)
5. FREQUÊNCIA: quantas vezes ao dia
6. ALERTAS: algum cuidado especial

Seja direto e prático. Máximo 6 linhas.""",
            file_contents=[image_content]
        )
        
        analysis = await chat.send_message(message)
        
        # Adicionar à entrada
        products_photos = entry.get("products_photos", [])
        products_photos.append(image_base64)
        
        products_analysis = entry.get("products_analysis", [])
        products_analysis.append(analysis)
        
        await db.daily_entries.update_one(
            {"id": entry_id},
            {"$set": {
                "products_photos": products_photos,
                "products_analysis": products_analysis,
                "updated_at": datetime.utcnow()
            }}
        )
        
        return {"analysis": analysis}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao analisar produto: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/admin/stats")
async def get_admin_stats():
    """Retorna estatísticas para o dashboard admin"""
    try:
        total_users = await db.users.count_documents({})
        premium_users = await db.users.count_documents({"is_premium": True})
        subscribers = await db.users.count_documents({"is_subscriber": True})
        total_analyses = await db.facial_analyses.count_documents({})
        total_quiz = await db.quiz_results.count_documents({})
        
        # Pegar últimos usuários
        recent_users = await db.users.find().sort("created_at", -1).limit(10).to_list(10)
        
        return {
            "total_users": total_users,
            "premium_users": premium_users,
            "subscribers": subscribers,
            "total_analyses": total_analyses,
            "total_quiz_completions": total_quiz,
            "conversion_premium": round((premium_users / total_users * 100) if total_users > 0 else 0, 2),
            "conversion_subscription": round((subscribers / premium_users * 100) if premium_users > 0 else 0, 2),
            "recent_users": [User(**u) for u in recent_users]
        }
        
    except Exception as e:
        logger.error(f"Erro nas estatísticas: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
