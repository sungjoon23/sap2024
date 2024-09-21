from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import requests
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import io

app = FastAPI()

KAMIS_API_URL = "http://www.kamis.or.kr/service/price/xml.do"
API_KEY = "53824154-ba14-4a94-b2b2-d7b3cb7585b3"
API_ID = "guaum0817@gmail.com"

templates = Jinja2Templates(directory="templates")

ITEMS = ["사과", "배", "포도", "감자", "고구마", "호박", "양파", "당근", "대파"]

@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    items_options = "".join([f'<option value="{item}">{item}</option>' for item in ITEMS])
    return templates.TemplateResponse("index.html", {"request": request, "items_options": items_options})

@app.post("/search", response_class=HTMLResponse)
async def search_item(item: str = Form(...)):
    params = {
        "action": "dailySalesList",
        "p_cert_key": API_KEY,
        "p_cert_id": API_ID,
        "p_returntype": "xml"
    }

    response = requests.get(KAMIS_API_URL, params=params)

    if response.status_code == 200:
        content = response.text

        root = ET.fromstring(content)

        matching_items = []
        for item_element in root.findall(".//item"):
            product_name = item_element.find("item_name").text if item_element.find("item_name") is not None else "N/A"
            price = item_element.find("dpr1").text if item_element.find("dpr1") is not None else "N/A"
            category_name = item_element.find("category_name").text if item_element.find("category_name") is not None else "N/A"
            unit = item_element.find("unit").text if item_element.find("unit") is not None else "N/A"
            lastest_date = item_element.find("lastest_date").text if item_element.find("lastest_date") is not None else "N/A"
            day1 = item_element.find("day1").text if item_element.find("day1") is not None else "N/A"
            day2 = item_element.find("dpr2").text if item_element.find("dpr2") is not None else "N/A"
            day3 = item_element.find("dpr3").text if item_element.find("dpr3") is not None else "N/A"
            day4 = item_element.find("dpr4").text if item_element.find("dpr4") is not None else "N/A"
            direction = item_element.find("direction").text if item_element.find("direction") is not None else "N/A"
            value = item_element.find("value").text if item_element.find("value") is not None else "N/A"
            product_cls_code = item_element.find("product_cls_code").text if item_element.find("product_cls_code") is not None else "N/A"

            if product_cls_code == "01":
                trade_type = "소매"
            elif product_cls_code == "02":
                trade_type = "도매"
            else:
                trade_type = "알 수 없음"

            if direction == "0":
                direction_text = "가격 하락"
            elif direction == "1":
                direction_text = "가격 상승"
            elif direction == "2":
                direction_text = "등락 없음"
            else:
                direction_text = "알 수 없음"

            if item.lower() in product_name.lower():
                matching_items.append(f"""
                    <strong>품목명:</strong> {product_name} <br>
                    <strong>가격:</strong> {price} <br>
                    <strong>부류명:</strong> {category_name} <br>
                    <strong>단위:</strong> {unit} <br>
                    <strong>최신 조사일자:</strong> {day1} <br>
                    <strong>1일 전 가격:</strong> {day2} <br>
                    <strong>1개월 전 가격:</strong> {day3} <br>
                    <strong>1년 전 가격:</strong> {day4} <br>
                    <strong>도/소매 구분:</strong> {trade_type} <br>
                    <strong>등락 여부:</strong> {direction_text} <br>
                    <strong>등락율:</strong> {value}% <br><br>
                """)

        if matching_items:
            results = "<br>".join(matching_items)
        else:
            results = f"'{item}'에 대한 결과를 찾을 수 없습니다."

        return results
    else:
        return f"Error: Unable to fetch data from KAMIS API. Status code: {response.status_code}"

@app.get("/graph", response_class=StreamingResponse)
async def graph(item: str):
    # API 호출
    params = {
        "action": "dailySalesList",
        "p_cert_key": API_KEY,
        "p_cert_id": API_ID,
        "p_returntype": "xml"
    }

    response = requests.get(KAMIS_API_URL, params=params)

    if response.status_code == 200:
        content = response.text
        root = ET.fromstring(content)

        prices_list = []
        product_names = []
        labels = ["Today", "Yesterday", "Month", "Year"]


        for item_element in root.findall(".//item"):
            product_name = item_element.find("item_name").text if item_element.find("item_name") is not None else "N/A"


            if item.lower() in product_name.lower():
                day1 = item_element.find("day1").text if item_element.find("day1") is not None else None
                day2 = item_element.find("dpr2").text if item_element.find("dpr2") is not None else None
                day3 = item_element.find("dpr3").text if item_element.find("dpr3") is not None else None
                day4 = item_element.find("dpr4").text if item_element.find("dpr4") is not None else None
                current_price = item_element.find("dpr1").text if item_element.find("dpr1") is not None else None


                prices = [current_price, day2, day3, day4]
                prices_valid = [int(p.replace(",", "")) if p and p not in ["N/A", "None"] else None for p in prices]

                if any(prices_valid):
                    product_names.append(product_name)
                    prices_list.append(prices_valid)

        if prices_list:
            fig, axes = plt.subplots(len(prices_list), 1, figsize=(6, 4 * len(prices_list)))

            if len(prices_list) == 1:
                axes = [axes]

            for i, prices in enumerate(prices_list):
                valid_prices = [p for p in prices if p is not None]
                valid_labels = [labels[j] for j, p in enumerate(prices) if p is not None]  
                if valid_prices:
                    axes[i].scatter(valid_labels, valid_prices, color='red', s=100)
                    axes[i].set_xlabel('Period')
                    axes[i].set_ylabel('Price')

            plt.tight_layout()

            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            return StreamingResponse(img, media_type="image/png")

        return HTMLResponse(content="유효한 가격 데이터가 없습니다. 그래프를 그릴 수 없습니다.", status_code=404)

    return HTMLResponse(content="API 요청에 실패했습니다.", status_code=response.status_code)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)