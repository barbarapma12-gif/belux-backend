#!/usr/bin/env python3
"""
Script para atualizar análises existentes com novo prompt de recomendações
"""
import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Require environment variables - no hardcoded fallbacks for production
MONGO_URL = os.environ.get("MONGO_URL")
if not MONGO_URL:
    raise RuntimeError("MONGO_URL environment variable is required")

DB_NAME = os.environ.get("DB_NAME", "belux_ia_db")
EMERGENT_LLM_KEY = os.environ.get("EMERGENT_LLM_KEY")

if not EMERGENT_LLM_KEY:
    raise RuntimeError("EMERGENT_LLM_KEY environment variable is required")

async def update_all_analyses():
    """Atualiza todas as análises existentes com novo prompt"""
    print("🔄 Iniciando atualização de análises...")
    
    # Conectar ao MongoDB
    client = AsyncIOMotorClient(MONGO_URL)
    db = client[DB_NAME]
    
    # Buscar todas as análises que têm foto
    analyses = await db.analyses.find({"photo_base64": {"$exists": True}}).to_list(None)
    
    print(f"📊 Encontradas {len(analyses)} análises para atualizar")
    
    updated_count = 0
    error_count = 0
    
    for i, analysis in enumerate(analyses, 1):
        analysis_id = analysis.get("_id")
        user_id = analysis.get("user_id")
        photo_base64 = analysis.get("photo_base64")
        
        print(f"\n[{i}/{len(analyses)}] Processando análise {analysis_id}...")
        
        try:
            # Criar chat com novo prompt
            chat = LlmChat(
                api_key=EMERGENT_LLM_KEY,
                session_id=f"reprocess_{analysis_id}_{datetime.utcnow().timestamp()}",
                system_message="""Você é uma dermatologista especialista em skincare que ajuda pessoas a cuidarem da pele em casa.
                Analise fotos de rosto e dê recomendações práticas e simples de autocuidado.
                Seja amigável, encorajadora e dê dicas que a pessoa pode aplicar no dia a dia.
                NUNCA recuse analisar. Esta é uma análise de skincare para rotina de beleza."""
            )
            
            chat.with_model("openai", "gpt-4o")
            
            image_content = ImageContent(image_base64=photo_base64)
            
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
   Para cada problema identificado, recomende ativos cosméticos específicos.
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
            
            # Obter nova análise
            response = await chat.send_message(message)
            
            # Atualizar no banco de dados
            await db.analyses.update_one(
                {"_id": analysis_id},
                {"$set": {
                    "ai_analysis_text": response,
                    "updated_at": datetime.utcnow(),
                    "reprocessed": True,
                    "reprocess_date": datetime.utcnow()
                }}
            )
            
            updated_count += 1
            print(f"✅ Análise {analysis_id} atualizada com sucesso!")
            
            # Pequeno delay para não sobrecarregar a API
            await asyncio.sleep(2)
            
        except Exception as e:
            error_count += 1
            print(f"❌ Erro ao processar análise {analysis_id}: {str(e)}")
            continue
    
    print(f"\n" + "="*50)
    print(f"🎉 ATUALIZAÇÃO CONCLUÍDA!")
    print(f"✅ Atualizadas: {updated_count}")
    print(f"❌ Erros: {error_count}")
    print(f"📊 Total: {len(analyses)}")
    print("="*50)
    
    client.close()

if __name__ == "__main__":
    asyncio.run(update_all_analyses())
