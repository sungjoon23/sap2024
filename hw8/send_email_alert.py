import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

def send_email_alert():
    # 이메일 구성
    message = Mail(
        from_email='guaum12@naver.com',  # 발신자 이메일 주소
        to_emails='guaum12@naver.com',   # 수신자 이메일 주소
        subject='Hourly AWS Weather Data Alert',
        html_content="<strong>데이터가 업데이트되었습니다.</strong><br> AWS 날씨 데이터 확인해."
    )

    try:
        api_key = os.getenv('SENDGRID_API_KEY')
        
        # 디버깅을 위한 확인 (키가 없는 경우 메시지 출력)
        if not api_key:
            print("Error: SENDGRID_API_KEY 환경 변수가 설정되지 않았습니다.")
            return

        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        print("이메일이 전송되었습니다:", response.status_code)
    except Exception as e:
        print(f"이메일 전송 실패: {e}")

if __name__ == "__main__":
    send_email_alert()
