import json
import os
import re
import logging
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI

app = Flask(__name__, static_folder=".", static_url_path="")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COMPANY_NAME = "Zhejiang Huapu New Materials Co., Ltd."
COMPANY_NAME_ZH = "浙江华普新材有限公司"
COMPANY_INTRO_EN = """We are a leading manufacturer of prepainted steel coils (PPGI), galvalume steel coils (GL), galvanized steel coils (GI), and cold-rolled steel products. With advanced production lines and strict quality control, we serve customers in over 50 countries across construction, home appliance, and automotive industries."""
COMPANY_INTRO_ZH = """我们是预涂镀锌钢卷(PPGI)、镀铝锌钢卷(GL)、镀锌钢卷(GI)和冷轧钢产品的领先制造商。拥有先进的生产线和严格的质量控制，产品远销50多个国家，服务于建筑、家电和汽车行业。"""

LANGUAGE_MAP = {
    "en": "English",
    "zh": "Chinese (中文)",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "ar": "Arabic (العربية)",
    "ru": "Russian (Русский)",
}

client = None
api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
api_base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
model = os.environ.get("LLM_MODEL") or "deepseek-chat"

if api_key:
    client = OpenAI(api_key=api_key, base_url=api_base)
    logger.info(f"LLM client initialized: model={model}, base_url={api_base}")
else:
    logger.warning("No API key found. Set OPENAI_API_KEY or DEEPSEEK_API_KEY environment variable.")


SYSTEM_PROMPT = """You are a professional business development specialist working for {company_name}.

Company Background:
- Company: {company_name} ({company_name_zh})
- Industry: Steel coil manufacturing (PPGI, Galvalume, Galvanized, Cold-rolled steel)
- {company_intro}

Your task: Write a compelling, professional cold outreach / business development email to a potential customer. The email should:
1. Have a clear, engaging subject line
2. Be personalized with the recipient's name and company
3. Introduce our company and product concisely
4. Highlight key value propositions (quality, competitive pricing, certifications, global experience)
5. Include a clear call-to-action (request for inquiry, video call, or sample request)
6. Be professional but warm, not pushy
7. Keep it concise (200-350 words)
8. Match the requested language precisely

IMPORTANT: Output ONLY the email content. Do NOT include explanations, notes, or markdown formatting around the email. Start directly with the subject line.

Format:
Subject: [subject line]

[email body]

Best regards,
[Name]
{company_name}
[Contact placeholder]"""


def generate_fallback_letter(customer_name, customer_company, product_name, language):
    """Fallback template-based letter when LLM is unavailable."""
    if language == "zh":
        return f"""Subject: {product_name} 优质供应商 - {COMPANY_NAME_ZH}

尊敬的 {customer_name}，

您好！我是来自{COMPANY_NAME_ZH}的销售代表。

我们了解到贵公司 {customer_company} 在相关领域的业务发展，特此联系您，希望有机会为贵公司提供高品质的{product_name}产品。

作为国内领先的涂层钢卷制造商，我们拥有先进的生产设备和完善的质量管理体系。我们的产品已出口至50多个国家，广泛应用于建筑、家电和汽车行业。

我们的优势：
- 高品质产品，通过ISO、SGS等国际认证
- 极具竞争力的价格
- 灵活的起订量和交期
- 专业的售后支持

如您对我们的{product_name}感兴趣，欢迎随时联系我。我们可以提供详细的产品规格书和报价单，也可以安排样品寄送。

期待与贵公司建立长期合作关系！

此致
敬礼

{COMPANY_NAME_ZH}
电话/WhatsApp: [请联系我们]
邮箱: [请联系我们]"""
    else:
        return f"""Subject: Reliable Supplier of {product_name} - {COMPANY_NAME}

Dear {customer_name},

Greetings from {COMPANY_NAME}!

I hope this email finds you well. I am reaching out because I believe there could be a great opportunity for us to collaborate with {customer_company}.

We are a premier manufacturer of {product_name} and other coated steel products, with over a decade of experience serving clients across 50+ countries. Our production facilities are equipped with state-of-the-art lines, and we hold ISO, SGS, and other international certifications.

Why choose us:
- Premium quality products with rigorous QC
- Highly competitive pricing
- Flexible MOQ and on-time delivery
- Dedicated after-sales support

I would be happy to provide you with detailed product specifications, certifications, and a competitive quotation. We can also arrange samples for your evaluation.

Would you be open to a brief call or video meeting to discuss further?

Looking forward to the possibility of working together.

Best regards,
{COMPANY_NAME}
Tel/WhatsApp: [Contact Us]
Email: [Contact Us]"""


def search_and_generate(customer_name, customer_company, product_name, language):
    """Generate development letter using LLM."""
    if client is None:
        logger.info("No LLM client available, using fallback template")
        return generate_fallback_letter(customer_name, customer_company, product_name, language)

    lang_name = LANGUAGE_MAP.get(language, "English")

    user_prompt = f"""Please write a business development email with the following details:

Recipient Name: {customer_name}
Recipient Company: {customer_company}
Our Product to Promote: {product_name}
Language: {lang_name}

Write the email in {lang_name}. Make it personalized, professional, and effective."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(
                    company_name=COMPANY_NAME,
                    company_name_zh=COMPANY_NAME_ZH,
                    company_intro=COMPANY_INTRO_EN
                )},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            max_tokens=1500,
        )
        letter = response.choices[0].message.content.strip()
        logger.info(f"LLM generated letter, length={len(letter)}")
        return letter
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return generate_fallback_letter(customer_name, customer_company, product_name, language)


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "Invalid JSON"}), 400

        customer_name = data.get("customer_name", "").strip()
        customer_company = data.get("customer_company", "").strip()
        product_name = data.get("product_name", "").strip()
        language = data.get("language", "en").strip()

        if not all([customer_name, customer_company, product_name]):
            return jsonify({"success": False, "error": "请填写所有字段 / Please fill in all fields"}), 400

        if language not in LANGUAGE_MAP:
            language = "en"

        logger.info(f"Generating letter for: {customer_name} @ {customer_company}, product={product_name}, lang={language}")

        letter = search_and_generate(customer_name, customer_company, product_name, language)

        return jsonify({"success": True, "letter": letter})

    except Exception as e:
        logger.exception("Generate endpoint error")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    logger.info("Starting 开发信生成器 server on http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=True)
