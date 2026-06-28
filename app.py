import json
import os
import re
import logging
from pathlib import Path
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

# --- Settings (persisted to JSON file) ---
SETTINGS_FILE = Path(__file__).parent / "settings.json"

def load_settings():
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except:
            pass
    return {}

def save_settings(data):
    SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def get_llm_client():
    """Create OpenAI client from settings, fallback to env vars."""
    settings = load_settings()
    api_key = settings.get("api_key") or os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    api_base = settings.get("base_url") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    model = settings.get("model") or os.environ.get("LLM_MODEL") or "deepseek-chat"
    if not api_key:
        return None, model
    return OpenAI(api_key=api_key, base_url=api_base), model


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

    elif language == "es":
        return f"""Subject: Proveedor confiable de {product_name} - {COMPANY_NAME}

Estimado/a {customer_name},

Reciba un cordial saludo desde {COMPANY_NAME}. Me pongo en contacto con usted porque creemos que existe una gran oportunidad de colaboración con {customer_company}.

Somos un fabricante líder de {product_name} y otros productos de acero recubierto, con más de una década de experiencia atendiendo clientes en más de 50 países. Nuestras instalaciones cuentan con líneas de producción de última generación y contamos con certificaciones internacionales ISO, SGS, entre otras.

Nuestras ventajas:
- Productos de alta calidad con control de calidad riguroso
- Precios altamente competitivos
- MOQ flexible y entrega puntual
- Soporte postventa dedicado

Estaré encantado de proporcionarle especificaciones detalladas del producto, certificaciones y una cotización competitiva. También podemos enviar muestras para su evaluación.

¿Le parecería bien una breve llamada o videollamada para conversar más al respecto?

Quedo a la espera de su respuesta.

Atentamente,
{COMPANY_NAME}
Tel/WhatsApp: [Contáctenos]
Email: [Contáctenos]"""

    elif language == "fr":
        return f"""Subject: Fournisseur fiable de {product_name} - {COMPANY_NAME}

Cher/Chère {customer_name},

Je vous adresse mes salutations depuis {COMPANY_NAME}. Je vous contacte car nous croyons qu'il existe une belle opportunité de collaboration avec {customer_company}.

Nous sommes un fabricant de premier plan de {product_name} et d'autres produits en acier revêtu, avec plus de dix ans d'expérience au service de clients dans plus de 50 pays. Nos installations sont équipées de lignes de production de pointe et nous détenons les certifications internationales ISO, SGS, entre autres.

Nos avantages :
- Produits de haute qualité avec un contrôle qualité rigoureux
- Prix très compétitifs
- MOQ flexible et livraison ponctuelle
- Support après-vente dédié

Je serais ravi de vous fournir des spécifications détaillées du produit, des certifications et un devis compétitif. Nous pouvons également envoyer des échantillons pour votre évaluation.

Seriez-vous disponible pour un bref appel ou une visioconférence afin d'en discuter davantage ?

Dans l'attente de votre réponse.

Cordialement,
{COMPANY_NAME}
Tél/WhatsApp: [Contactez-nous]
Email: [Contactez-nous]"""

    elif language == "ar":
        return f"""Subject: مورد موثوق لـ {product_name} - {COMPANY_NAME}

السيد/السيدة {customer_name} المحترم/ة،

تحية طيبة من {COMPANY_NAME}. نتواصل معكم لأننا نعتقد أن هناك فرصة رائعة للتعاون مع شركة {customer_company}.

نحن من الشركات الرائدة في تصنيع {product_name} ومنتجات الفولاذ المطلية الأخرى، مع أكثر من عشر سنوات من الخبرة في خدمة العملاء في أكثر من 50 دولة. تتميز مرافقنا بخطوط إنتاج متطورة ونحمل شهادات دولية مثل ISO وSGS وغيرها.

مميزاتنا:
- منتجات عالية الجودة مع رقابة صارمة
- أسعار تنافسية للغاية
- حد أدنى مرن للطلب وتسليم في الموعد
- دعم مخصص لخدمة ما بعد البيع

يسعدنا تزويدكم بالمواصفات التفصيلية للمنتج والشهادات وعرض أسعار تنافسي. يمكننا أيضاً إرسال عينات للتقييم.

هل يمكن ترتيب مكالمة قصيرة أو اجتماع عبر الفيديو لمناقشة المزيد؟

نتطلع للرد عليكم.

مع أطيب التحيات،
{COMPANY_NAME}
هاتف/واتساب: [اتصل بنا]
البريد الإلكتروني: [اتصل بنا]"""

    elif language == "ru":
        return f"""Subject: Надёжный поставщик {product_name} - {COMPANY_NAME}

Уважаемый/ая {customer_name},

Приветствуем вас из {COMPANY_NAME}! Мы обращаемся к вам, поскольку считаем, что существует отличная возможность для сотрудничества с компанией {customer_company}.

Мы являемся ведущим производителем {product_name} и другой продукции из стали с покрытием, с более чем десятилетним опытом обслуживания клиентов в более чем 50 странах. Наши производственные мощности оснащены современными линиями, и мы имеем международные сертификаты ISO, SGS и другие.

Наши преимущества:
- Высококачественная продукция с тщательным контролем качества
- Высококонкурентные цены
- Гибкий минимальный заказ и своевременная доставка
- Специализированная послепродажная поддержка

Буду рад предоставить вам подробные спецификации продукции, сертификаты и конкурентное коммерческое предложение. Также можем организовать отправку образцов для оценки.

Не могли бы вы уделить время для краткого звонка или видеовстречи для обсуждения деталей?

С нетерпением жду вашего ответа.

С уважением,
{COMPANY_NAME}
Тел/WhatsApp: [Свяжитесь с нами]
Email: [Свяжитесь с нами]"""

    else:  # English default
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
    client, llm_model = get_llm_client()
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
            model=llm_model,
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
        # 保底执行模板生成，确保用户体验不受影响(即使没有大模型连接上来)
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


@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    s = load_settings()
    return jsonify({
        "api_key": s.get("api_key", ""),
        "base_url": s.get("base_url", "https://api.deepseek.com"),
        "model": s.get("model", "deepseek-chat"),
        "smtp_server": s.get("smtp_server", "smtp.qq.com"),
        "smtp_port": s.get("smtp_port", "465"),
        "smtp_user": s.get("smtp_user", "512464829@qq.com"),
        "smtp_password": s.get("smtp_password", ""),
    })

@app.route("/api/settings", methods=["PUT"])
def api_save_settings():
    data = request.get_json()
    s = load_settings()
    s.update({
        "api_key": data.get("api_key", s.get("api_key", "")).strip(),
        "base_url": data.get("base_url", s.get("base_url", "https://api.deepseek.com")).strip(),
        "model": data.get("model", s.get("model", "deepseek-chat")).strip(),
        "smtp_server": data.get("smtp_server", s.get("smtp_server", "")).strip(),
        "smtp_port": data.get("smtp_port", s.get("smtp_port", "465")).strip(),
        "smtp_user": data.get("smtp_user", s.get("smtp_user", "")).strip(),
        "smtp_password": data.get("smtp_password", s.get("smtp_password", "")).strip(),
    })
    save_settings(s)
    return jsonify({"status": "ok"})


@app.route("/api/send-email", methods=["POST"])
def api_send_email():
    """Send the generated letter via SMTP."""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    data = request.get_json()
    to_email = data.get("to_email", "").strip()
    subject = data.get("subject", "").strip()
    body = data.get("body", "").strip()

    if not to_email or not body:
        return jsonify({"success": False, "error": "收件人邮箱和邮件内容不能为空"}), 400

    s = load_settings()
    smtp_server = s.get("smtp_server", "")
    smtp_port = int(s.get("smtp_port", "465") or "465")
    smtp_user = s.get("smtp_user", "")
    smtp_password = s.get("smtp_password", "")

    if not smtp_server or not smtp_user or not smtp_password:
        return jsonify({"success": False, "error": "请先点右上角 ⚙ 设置，配置你的发件邮箱（SMTP服务器、账号、授权码）"}), 400

    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_user
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=15) as server:
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_server, smtp_port, timeout=15) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)

        logger.info(f"Email sent to {to_email}")
        return jsonify({"success": True, "message": f"邮件已发送至 {to_email}"})

    except smtplib.SMTPAuthenticationError:
        return jsonify({"success": False, "error": "SMTP 认证失败，请检查邮箱账号和授权码"}), 401
    except Exception as e:
        logger.error(f"Send email failed: {e}")
        return jsonify({"success": False, "error": f"发送失败: {str(e)}"}), 500


if __name__ == "__main__":
    logger.info("Starting 开发信生成器 server on http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=True)
