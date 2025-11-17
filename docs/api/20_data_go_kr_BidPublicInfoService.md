---
layout: default
title: "데이타포탈 - 혁신장터 최종제안요청서 교부 첨부파일정보 조회 API"
description: "데이타포탈 - 혁신장터 최종제안요청서 교부 첨부파일정보 조회 API"
date: 2025-11-17
author: "김명환"
cache-control: no-cache
expires: 0
pragma: no-cache
---

# 혁신장터 최종제안요청서 교부 첨부파일정보 조회 API

## 1. API 개요

### 1.1. 서비스 정보
- **API명**: getBidPblancListPPIFnlRfpIssAtchFileInfo
- **서비스**: 나라장터 입찰공고정보서비스 (BidPublicInfoService)
- **용도**: 혁신장터 최종제안요청서 교부 첨부파일 정보 조회

### 1.2. 설명
낙찰자결정방법이 **경쟁적 대화에 의한 낙찰자 선정 방법**일 경우, 검색조건에 공고게시일시, 교부일시, 입찰공고번호를 입력하여 혁신장터에서 교부된 최종제안요청서 첨부파일 정보를 조회합니다.

### 1.3. 성능 정보
- **최대 메시지 사이즈**: 4000 bytes
- **평균 응답 시간**: 500 ms
- **초당 최대 트랜잭션**: 30 tps

## 2. 엔드포인트

### 2.1. Base URL
```
https://apis.data.go.kr/1230000/ad/BidPublicInfoService/getBidPblancListPPIFnlRfpIssAtchFileInfo

https://apis.data.go.kr/1230000/ad/BidPublicInfoService/getBidPblancListPPIFnlRfpIssAtchFileInfo?serviceKey=8fea481d8c22a53e6baa58ac84350a16577b68d4ba8ee4b2c5ce3057db4bbbfa&pageNo=1&numOfRows=10&inqryDiv=1&type=json&bidNtceNo=R25BK01156560&inqryBgnDt=202506010000&inqryEndDt=202507010000
```

### 2.2. HTTP Method
- GET

### 2.3. 프로토콜
- REST (GET)

## 3. 요청 파라미터

### 3.1. 공통 파라미터

| 파라미터명 | 한글명 | 타입 | 필수 | 설명 |
|-----------|--------|------|------|------|
| serviceKey | 서비스키 | String(400) | 필수 | 공공데이터포탈에서 발급받은 인증키 |
| pageNo | 페이지번호 | Integer(4) | 필수 | 페이지 번호 (기본값: 1) |
| numOfRows | 한 페이지 결과 수 | Integer(4) | 필수 | 페이지당 결과 수 (기본값: 10) |
| type | 응답 타입 | String(4) | 선택 | 'json' 지정 시 JSON 형식 응답 (미지정 시 XML) |

### 3.2. 조회 파라미터

| 파라미터명 | 한글명 | 타입 | 필수 | 설명 |
|-----------|--------|------|------|------|
| inqryDiv | 조회구분 | String(1) | 필수 | 1: 공고게시일시<br>2: 교부일시<br>3: 입찰공고번호 |
| inqryBgnDt | 조회시작일시 | String(12) | 조건부 필수 | YYYYMMDDHHMM 형식<br>(inqryDiv가 1 또는 2일 때 필수)<br>**최대 범위: 1개월** |
| inqryEndDt | 조회종료일시 | String(12) | 조건부 필수 | YYYYMMDDHHMM 형식<br>(inqryDiv가 1 또는 2일 때 필수)<br>**최대 범위: 1개월** |
| bidNtceNo | 입찰공고번호 | String(40) | 조건부 필수 | 입찰공고번호<br>(inqryDiv가 3일 때 필수) |

### 3.3. 조회 범위 제한
- **중요**: inqryDiv가 1 또는 2일 경우, 조회 기간은 **최대 1개월**로 제한됩니다.
- 1개월을 초과하는 기간으로 조회 시 오류가 발생할 수 있습니다.

### 3.4. 요청 예시

#### 공고게시일시 기준 조회
```
https://apis.data.go.kr/1230000/ad/BidPublicInfoService/getBidPblancListPPIFnlRfpIssAtchFileInfo?serviceKey=8fea481d8c22a53e6baa58ac84350a16577b68d4ba8ee4b2c5ce3057db4bbbfa&pageNo=1&numOfRows=10&inqryDiv=1&type=json&inqryBgnDt=202506010000&inqryEndDt=202507010000
```

#### 입찰공고번호 기준 조회
```
https://apis.data.go.kr/1230000/ad/BidPublicInfoService/getBidPblancListPPIFnlRfpIssAtchFileInfo?serviceKey=8fea481d8c22a53e6baa58ac84350a16577b68d4ba8ee4b2c5ce3057db4bbbfa&pageNo=1&numOfRows=10&inqryDiv=3&type=json&bidNtceNo=R25BK00900354
```

## 4. 응답 구조

### 4.1. 응답 형식

```json
{
  "response": {
    "header": {
      "resultCode": "string",
      "resultMsg": "string"
    },
    "body": {
      "items": [
        { 첨부파일 정보 객체 }
      ],
      "numOfRows": integer,
      "pageNo": integer,
      "totalCount": integer
    }
  }
}
```

### 4.2. Header 필드

| 필드명 | 한글명 | 타입 | 설명 |
|--------|--------|------|------|
| resultCode | 결과코드 | String(2) | "00": 정상 처리<br>기타: 오류 코드 |
| resultMsg | 결과메시지 | String(50) | 처리 결과 메시지 |

### 4.3. Body 필드

| 필드명 | 한글명 | 타입 | 설명 |
|--------|--------|------|------|
| numOfRows | 한 페이지 결과 수 | Integer | 페이지당 결과 수 |
| pageNo | 페이지 번호 | Integer | 현재 페이지 번호 |
| totalCount | 전체 결과 수 | Integer | 전체 데이터 건수 |
| items | 첨부파일 목록 | Array | 첨부파일 정보 배열 |

### 4.4. Items 필드 (첨부파일 상세 정보)

| 필드명 | 한글명 | 타입 | 필수 | 설명 |
|--------|--------|------|------|------|
| bidNtceNo | 입찰공고번호 | String(40) | 필수 | 입찰공고 관리번호<br>차세대 나라장터:<br>R+년도(2)+단계구분(2)+순번(8)<br>총 13자리 |
| bidNtceOrd | 입찰공고차수 | String(3) | 필수 | 해당 입찰공고에 대한<br>재공고 및 재입찰 등이<br>발생되었을 경우 증가하는 수 |
| issDt | 교부일시 | String(19) | 선택 | 혁신장터 제안서 교부일시<br>YYYY-MM-DD HH:MM:SS 형식 |
| bsnsDivNm | 업무구분명 | String(30) | 필수 | 입찰업무 구분:<br>- 물품<br>- 용역<br>- 공사<br>- 외자 |
| atchSno | 첨부순번 | String(7) | 선택 | 공고의 최종 제안요청서<br>교부 첨부파일 순번 |
| atchDocDivNm | 첨부문서구분명 | String(50) | 선택 | 공고의 최종 제안요청서<br>교부 첨부파일 문서구분명<br>예: "최종제안요청서 교부안내서",<br>"최종제안요청서" |
| atchFileNm | 첨부파일명 | String(200) | 선택 | 공고의 최종 제안요청서<br>교부 첨부파일명 |
| atchFileUrl | 첨부파일URL | String(800) | 선택 | 공고의 최종 제안요청서<br>교부 첨부파일 다운로드 URL |

### 4.5. 입찰공고번호 체계

차세대 나라장터 번호 체계:
- **형식**: R + 년도(2자리) + 단계구분(2자리) + 순번(8자리) = 총 13자리
- **단계구분 코드**:
  - BK: 입찰
  - TA: 계약
  - DD: 발주계획
  - BD: 사전규격

예시: `R25BK00900354`
- R: 차세대 시스템 식별자
- 25: 2025년
- BK: 입찰 단계
- 00900354: 순번

## 5. 응답 예시

### 5.1. 성공 응답 (JSON)

```json
{
  "response": {
    "header": {
      "resultCode": "00",
      "resultMsg": "정상"
    },
    "body": {
      "items": [
        {
          "bidNtceNo": "R25BK00900354",
          "bidNtceOrd": "000",
          "issDt": "2025-06-25 13:43:14",
          "bsnsDivNm": "물품",
          "atchSno": "1",
          "atchDocDivNm": "최종제안요청서 교부안내서",
          "atchFileNm": "장당중학교 방송장비 구매 및 설치 제안서 제출안내 공고.hwpx",
          "atchFileUrl": "https://www.g2b.go.kr/pn/pnp/pnpe/UntyAtchFile/downloadFile.do?bidPbancNo=R25BK00900354&bidPbancOrd=000&fileType=LSTG&fileSeq=1&prcmBsneSeCd=01"
        },
        {
          "bidNtceNo": "R25BK00900354",
          "bidNtceOrd": "000",
          "issDt": "2025-06-25 13:43:14",
          "bsnsDivNm": "물품",
          "atchSno": "2",
          "atchDocDivNm": "최종제안요청서",
          "atchFileNm": "제안요청서.hwpx",
          "atchFileUrl": "https://www.g2b.go.kr/pn/pnp/pnpe/UntyAtchFile/downloadFile.do?bidPbancNo=R25BK00900354&bidPbancOrd=000&fileType=LSTD&fileSeq=2&prcmBsneSeCd=01"
        },
        {
          "bidNtceNo": "R25BK00909111",
          "bidNtceOrd": "000",
          "issDt": "2025-07-14 16:41:59",
          "bsnsDivNm": "물품",
          "atchSno": "1",
          "atchDocDivNm": "최종제안요청서 교부안내서",
          "atchFileNm": "안산국제비즈니스고 강당 방송장비 구매 및 설치 제안서 제출 안내.hwp",
          "atchFileUrl": "https://www.g2b.go.kr/pn/pnp/pnpe/UntyAtchFile/downloadFile.do?bidPbancNo=R25BK00909111&bidPbancOrd=000&fileType=LSTG&fileSeq=1&prcmBsneSeCd=01"
        },
        {
          "bidNtceNo": "R25BK00909111",
          "bidNtceOrd": "000",
          "issDt": "2025-07-14 16:41:59",
          "bsnsDivNm": "물품",
          "atchSno": "2",
          "atchDocDivNm": "최종제안요청서",
          "atchFileNm": "제안요청서.hwp",
          "atchFileUrl": "https://www.g2b.go.kr/pn/pnp/pnpe/UntyAtchFile/downloadFile.do?bidPbancNo=R25BK00909111&bidPbancOrd=000&fileType=LSTD&fileSeq=2&prcmBsneSeCd=01"
        }
      ],
      "numOfRows": 10,
      "pageNo": 1,
      "totalCount": 4
    }
  }
}
```

### 5.2. 성공 응답 (XML)

```xml
<response>
  <header>
    <resultCode>00</resultCode>
    <resultMsg>정상</resultMsg>
  </header>
  <body>
    <items>
      <item>
        <bidNtceNo>R25BK00734839</bidNtceNo>
        <bidNtceOrd>000</bidNtceOrd>
        <issDt>2025-05-16 17:07:21</issDt>
        <bsnsDivNm>물품</bsnsDivNm>
        <atchSno>1</atchSno>
        <atchDocDivNm>최종제안요청서 교부안내서</atchDocDivNm>
        <atchFileNm>붙임_1. 최종제안요청서 교부 안내서.hwp</atchFileNm>
        <atchFileUrl>https://www.g2b.go.kr/pn/pnp/pnpe/UntyAtchFile/downloadFile.do?bidPbancNo=R25BK00734839&bidPbancOrd=000&fileType=&fileSeq=1</atchFileUrl>
      </item>
      <item>
        <bidNtceNo>R25BK00734839</bidNtceNo>
        <bidNtceOrd>000</bidNtceOrd>
        <issDt>2025-05-16 17:07:21</issDt>
        <bsnsDivNm>물품</bsnsDivNm>
        <atchSno>2</atchSno>
        <atchDocDivNm>최종제안요청서</atchDocDivNm>
        <atchFileNm>최종제안요청서.pdf</atchFileNm>
        <atchFileUrl>https://www.g2b.go.kr/pn/pnp/pnpe/UntyAtchFile/downloadFile.do?bidPbancNo=R25BK00734839&bidPbancOrd=000&fileType=&fileSeq=2</atchFileUrl>
      </item>
      <item>
        <bidNtceNo>R25BK00734839</bidNtceNo>
        <bidNtceOrd>000</bidNtceOrd>
        <issDt>2025-05-16 17:07:21</issDt>
        <bsnsDivNm>물품</bsnsDivNm>
        <atchSno>3</atchSno>
        <atchDocDivNm>최종제안요청서</atchDocDivNm>
        <atchFileNm>최종제안요청서_제안관련 서식.hwp</atchFileNm>
        <atchFileUrl>https://www.g2b.go.kr/pn/pnp/pnpe/UntyAtchFile/downloadFile.do?bidPbancNo=R25BK00734839&bidPbancOrd=000&fileType=&fileSeq=3</atchFileUrl>
      </item>
    </items>
    <numOfRows>10</numOfRows>
    <pageNo>1</pageNo>
    <totalCount>3</totalCount>
  </body>
</response>
```

### 5.3. 오류 응답

```json
{
  "response": {
    "header": {
      "resultCode": "03",
      "resultMsg": "필수 파라미터 누락"
    }
  }
}
```

## 6. 에러 코드

| 코드 | 메시지 | 설명 |
|------|--------|------|
| 00 | 정상 | 정상 처리 |
| 01 | 어플리케이션 에러 | 응용 프로그램 오류 |
| 02 | 데이터베이스 에러 | 데이터베이스 오류 |
| 03 | 필수 파라미터 누락 | 필수 파라미터 미입력 |
| 04 | HTTP 에러 | HTTP 프로토콜 오류 |
| 05 | 서비스 연결 실패 | 서비스 연결 실패 |
| 10 | 잘못된 요청 파라미터 | 요청 파라미터 오류 |
| 11 | 유효하지 않은 인증키 | 서비스키 인증 실패 |
| 12 | 활용승인 미신청 | 서비스 미신청 |
| 99 | 기타 에러 | 기타 오류 |

## 7. 주의사항

### 7.1. 적용 대상
이 API는 **경쟁적 대화에 의한 낙찰자 선정 방법**이 적용된 입찰공고에만 해당합니다.
- 일반 입찰공고에는 최종제안요청서가 없으므로 데이터가 반환되지 않습니다.
- 혁신장터를 통해 진행되는 특정 공고에 한정됩니다.

### 7.2. 데이터 조회 제한
- **조회구분(inqryDiv)**에 따라 필수 파라미터가 달라집니다:
  - 공고게시일시/교부일시 조회 시: inqryBgnDt, inqryEndDt 필수
  - 입찰공고번호 조회 시: bidNtceNo 필수
- **조회 기간 제한**: inqryDiv가 1 또는 2일 때 최대 1개월

### 7.3. 첨부파일 다운로드
- atchFileUrl은 나라장터의 실제 파일 다운로드 링크입니다.
- URL은 일정 기간 후 변경될 수 있으므로 장기 보관용으로 적합하지 않습니다.
- 필요시 파일을 직접 다운로드하여 저장하는 것을 권장합니다.

### 7.4. 첨부파일 유형
일반적으로 다음과 같은 문서가 첨부됩니다:
1. **최종제안요청서 교부안내서**: 제안서 제출 안내 문서
2. **최종제안요청서**: 실제 제안요청서 본문
3. **제안관련 서식**: 제안서 작성에 필요한 양식

### 7.5. 응답 데이터 특성
- 빈 값인 필드는 빈 문자열("")로 반환되거나 필드 자체가 없을 수 있습니다.
- 하나의 입찰공고(bidNtceNo)에 여러 개의 첨부파일이 있을 수 있습니다.
- atchSno(첨부순번)로 첨부파일 순서를 구분합니다.

### 7.6. 성능 고려사항
- 한 번에 대량의 데이터 조회 시 pageNo와 numOfRows를 적절히 활용하세요.
- 1개월 단위로 데이터를 조회하고 필요시 여러 번 호출하세요.
- 불필요한 반복 호출을 피하고 적절한 캐싱 전략을 수립하세요.

## 8. 활용 예시

### 8.1. Python 예시

```python
import requests
import json
from datetime import datetime, timedelta

def get_bid_attachment_info(service_key, inqry_div, **kwargs):
    """
    혁신장터 최종제안요청서 첨부파일 조회
    
    Args:
        service_key (str): 공공데이터포탈 인증키
        inqry_div (str): 조회구분 (1: 공고게시일시, 2: 교부일시, 3: 입찰공고번호)
        **kwargs: 추가 파라미터
    
    Returns:
        dict: API 응답 데이터
    """
    url = "https://apis.data.go.kr/1230000/ad/BidPublicInfoService/getBidPblancListPPIFnlRfpIssAtchFileInfo"
    
    params = {
        "serviceKey": service_key,
        "pageNo": kwargs.get("pageNo", 1),
        "numOfRows": kwargs.get("numOfRows", 10),
        "type": "json",
        "inqryDiv": inqry_div
    }
    
    # 조회구분에 따른 필수 파라미터 추가
    if inqry_div in ["1", "2"]:
        params["inqryBgnDt"] = kwargs.get("inqryBgnDt")
        params["inqryEndDt"] = kwargs.get("inqryEndDt")
    elif inqry_div == "3":
        params["bidNtceNo"] = kwargs.get("bidNtceNo")
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API 호출 실패: {response.status_code}")

def download_attachment(file_url, save_path):
    """
    첨부파일 다운로드
    
    Args:
        file_url (str): 첨부파일 URL
        save_path (str): 저장 경로
    """
    response = requests.get(file_url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"파일 다운로드 완료: {save_path}")
    else:
        print(f"다운로드 실패: {response.status_code}")

# 사용 예시 1: 최근 1개월 데이터 조회
service_key = "your_service_key_here"

# 현재 시각과 1개월 전 시각 계산
end_dt = datetime.now()
start_dt = end_dt - timedelta(days=30)

result = get_bid_attachment_info(
    service_key=service_key,
    inqry_div="1",  # 공고게시일시 기준
    inqryBgnDt=start_dt.strftime("%Y%m%d%H%M"),
    inqryEndDt=end_dt.strftime("%Y%m%d%H%M")
)

# 결과 처리
if result["response"]["header"]["resultCode"] == "00":
    items = result["response"]["body"]["items"]
    
    for item in items:
        print(f"입찰공고번호: {item['bidNtceNo']}")
        print(f"첨부순번: {item['atchSno']}")
        print(f"문서구분: {item['atchDocDivNm']}")
        print(f"파일명: {item['atchFileNm']}")
        print(f"교부일시: {item['issDt']}")
        print(f"다운로드 URL: {item['atchFileUrl']}")
        print("-" * 80)
        
        # 첨부파일 다운로드 (선택사항)
        # download_attachment(item['atchFileUrl'], f"./downloads/{item['atchFileNm']}")
else:
    print(f"오류: {result['response']['header']['resultMsg']}")

# 사용 예시 2: 특정 입찰공고번호로 조회
result2 = get_bid_attachment_info(
    service_key=service_key,
    inqry_div="3",  # 입찰공고번호 기준
    bidNtceNo="R25BK00900354"
)

if result2["response"]["header"]["resultCode"] == "00":
    items = result2["response"]["body"]["items"]
    
    print(f"\n입찰공고번호 R25BK00900354의 첨부파일 목록:")
    for item in items:
        print(f"- [{item['atchDocDivNm']}] {item['atchFileNm']}")
else:
    print(f"오류: {result2['response']['header']['resultMsg']}")

# 사용 예시 3: 페이징 처리
def get_all_attachments(service_key, inqry_div, **kwargs):
    """
    모든 첨부파일 정보 조회 (페이징 자동 처리)
    
    Returns:
        list: 모든 첨부파일 정보 리스트
    """
    all_items = []
    page_no = 1
    
    while True:
        kwargs["pageNo"] = page_no
        result = get_bid_attachment_info(service_key, inqry_div, **kwargs)
        
        if result["response"]["header"]["resultCode"] != "00":
            break
        
        items = result["response"]["body"]["items"]
        if not items:
            break
        
        all_items.extend(items)
        
        # 전체 결과 수와 현재까지 조회한 수 비교
        total_count = result["response"]["body"]["totalCount"]
        if len(all_items) >= total_count:
            break
        
        page_no += 1
    
    return all_items

# 전체 데이터 조회
all_data = get_all_attachments(
    service_key=service_key,
    inqry_div="1",
    inqryBgnDt="202506010000",
    inqryEndDt="202507010000",
    numOfRows=100  # 페이지당 100건씩 조회
)

print(f"\n총 {len(all_data)}건의 첨부파일 조회 완료")
```

### 8.2. JavaScript 예시

```javascript
async function getBidAttachmentInfo(serviceKey, inqryDiv, options = {}) {
    const baseUrl = 'https://apis.data.go.kr/1230000/ad/BidPublicInfoService/getBidPblancListPPIFnlRfpIssAtchFileInfo';
    
    const params = new URLSearchParams({
        serviceKey: serviceKey,
        pageNo: options.pageNo || 1,
        numOfRows: options.numOfRows || 10,
        type: 'json',
        inqryDiv: inqryDiv
    });
    
    // 조회구분에 따른 필수 파라미터 추가
    if (['1', '2'].includes(inqryDiv)) {
        params.append('inqryBgnDt', options.inqryBgnDt);
        params.append('inqryEndDt', options.inqryEndDt);
    } else if (inqryDiv === '3') {
        params.append('bidNtceNo', options.bidNtceNo);
    }
    
    try {
        const response = await fetch(`${baseUrl}?${params}`);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('API 호출 오류:', error);
        throw error;
    }
}

// 사용 예시 1: 기본 조회
const serviceKey = 'your_service_key_here';

getBidAttachmentInfo(serviceKey, '1', {
    inqryBgnDt: '202506010000',
    inqryEndDt: '202507010000'
}).then(result => {
    if (result.response.header.resultCode === '00') {
        const items = result.response.body.items;
        
        items.forEach(item => {
            console.log(`입찰공고번호: ${item.bidNtceNo}`);
            console.log(`문서구분: ${item.atchDocDivNm}`);
            console.log(`파일명: ${item.atchFileNm}`);
            console.log(`다운로드 URL: ${item.atchFileUrl}`);
            console.log('-'.repeat(80));
        });
    } else {
        console.error(`오류: ${result.response.header.resultMsg}`);
    }
});

// 사용 예시 2: 입찰공고번호별 첨부파일 그룹핑
async function getAttachmentsByBidNo(serviceKey, inqryDiv, options) {
    const result = await getBidAttachmentInfo(serviceKey, inqryDiv, options);
    
    if (result.response.header.resultCode !== '00') {
        throw new Error(result.response.header.resultMsg);
    }
    
    const items = result.response.body.items;
    const grouped = {};
    
    items.forEach(item => {
        const bidNo = item.bidNtceNo;
        if (!grouped[bidNo]) {
            grouped[bidNo] = [];
        }
        grouped[bidNo].push(item);
    });
    
    return grouped;
}

// 그룹핑 예시 사용
getAttachmentsByBidNo(serviceKey, '1', {
    inqryBgnDt: '202506010000',
    inqryEndDt: '202507010000'
}).then(grouped => {
    Object.keys(grouped).forEach(bidNo => {
        console.log(`\n입찰공고번호: ${bidNo}`);
        grouped[bidNo].forEach(file => {
            console.log(`  - [${file.atchDocDivNm}] ${file.atchFileNm}`);
        });
    });
});

// 사용 예시 3: 첨부파일 다운로드 (Node.js 환경)
async function downloadAttachment(fileUrl, savePath) {
    const fs = require('fs');
    const https = require('https');
    
    return new Promise((resolve, reject) => {
        https.get(fileUrl, (response) => {
            const file = fs.createWriteStream(savePath);
            response.pipe(file);
            
            file.on('finish', () => {
                file.close();
                console.log(`파일 다운로드 완료: ${savePath}`);
                resolve();
            });
        }).on('error', (error) => {
            fs.unlink(savePath, () => {});
            reject(error);
        });
    });
}

// 모든 첨부파일 다운로드
async function downloadAllAttachments(serviceKey, bidNtceNo, downloadDir) {
    const result = await getBidAttachmentInfo(serviceKey, '3', {
        bidNtceNo: bidNtceNo
    });
    
    if (result.response.header.resultCode !== '00') {
        console.error(`오류: ${result.response.header.resultMsg}`);
        return;
    }
    
    const items = result.response.body.items;
    
    for (const item of items) {
        const savePath = `${downloadDir}/${item.atchFileNm}`;
        try {
            await downloadAttachment(item.atchFileUrl, savePath);
        } catch (error) {
            console.error(`다운로드 실패: ${item.atchFileNm}`, error);
        }
    }
}

// 다운로드 실행
downloadAllAttachments(serviceKey, 'R25BK00900354', './downloads');
```

### 8.3. 첨부파일 일괄 다운로드 스크립트 (Python)

```python
import os
import requests
from pathlib import Path

def download_all_bid_attachments(service_key, bid_ntce_no, save_dir):
    """
    특정 입찰공고의 모든 첨부파일 다운로드
    
    Args:
        service_key (str): API 서비스키
        bid_ntce_no (str): 입찰공고번호
        save_dir (str): 저장 디렉토리
    """
    # 저장 디렉토리 생성
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # API 호출
    result = get_bid_attachment_info(
        service_key=service_key,
        inqry_div="3",
        bidNtceNo=bid_ntce_no
    )
    
    if result["response"]["header"]["resultCode"] != "00":
        print(f"오류: {result['response']['header']['resultMsg']}")
        return
    
    items = result["response"]["body"]["items"]
    
    print(f"입찰공고번호 {bid_ntce_no}의 첨부파일 {len(items)}건 다운로드 시작")
    
    for idx, item in enumerate(items, 1):
        file_name = item['atchFileNm']
        file_url = item['atchFileUrl']
        save_path = os.path.join(save_dir, file_name)
        
        try:
            response = requests.get(file_url, timeout=30)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                print(f"[{idx}/{len(items)}] 다운로드 완료: {file_name}")
            else:
                print(f"[{idx}/{len(items)}] 다운로드 실패 ({response.status_code}): {file_name}")
        except Exception as e:
            print(f"[{idx}/{len(items)}] 오류 발생: {file_name} - {str(e)}")
    
    print(f"\n다운로드 완료. 저장 위치: {save_dir}")

# 사용 예시
service_key = "your_service_key_here"
download_all_bid_attachments(
    service_key=service_key,
    bid_ntce_no="R25BK00900354",
    save_dir="./bid_attachments/R25BK00900354"
)
```

## 9. 참고사항

### 9.1. 관련 API
- getBidPblancListInfoCnstwk: 공사 입찰공고 조회
- getBidPblancListInfoServc: 용역 입찰공고 조회
- getBidPblancListInfoFrgcpt: 외자 입찰공고 조회
- getBidPblancListInfoThng: 물품 입찰공고 조회

### 9.2. 혁신장터 (Public Procurement Innovation, PPI)
- **경쟁적 대화**: 복잡하고 혁신적인 조달에 적용되는 낙찰 방식
- **최종제안요청서**: 경쟁적 대화 과정을 거쳐 최종적으로 제안을 요청하는 문서
- 일반 입찰과 달리 대화를 통해 요구사항이 구체화되는 특징

### 9.3. 데이터 갱신 주기
- 최종제안요청서 교부 시 실시간 반영
- 첨부파일 등록 시 즉시 업데이트

### 9.4. 보안 및 저작권
- 다운로드한 첨부파일은 입찰 참여 목적으로만 사용하세요.
- 무단 배포 및 상업적 이용은 법적 문제가 발생할 수 있습니다.
- 파일 저장 시 개인정보보호 및 보안에 유의하세요.

### 9.5. 문의처
- **공공데이터포탈**: www.data.go.kr
- **나라장터**: www.g2b.go.kr
- **조달청 고객센터**: 1588-0800
- **혁신장터 문의**: 나라장터 고객센터를 통해 문의

## 10. FAQ

### Q1. 왜 조회 결과가 없나요?
A. 다음 사항을 확인하세요:
1. 해당 입찰공고가 경쟁적 대화 방식인지 확인
2. 최종제안요청서가 이미 교부되었는지 확인
3. 조회 기간이 1개월을 초과하지 않았는지 확인
4. 입찰공고번호가 정확한지 확인

### Q2. 첨부파일 다운로드가 안 됩니다.
A. 다음을 확인하세요:
1. atchFileUrl이 올바른지 확인
2. 네트워크 연결 상태 확인
3. URL 만료 여부 확인 (일정 시간 경과 시 URL 변경 가능)
4. 다운로드 권한 확인

### Q3. 조회 기간을 1개월 이상으로 설정하고 싶습니다.
A. 1개월 이상의 데이터를 조회하려면:
1. 기간을 1개월씩 나누어 여러 번 API 호출
2. 각 기간의 결과를 병합하여 사용
3. 예시: 3개월 데이터 = 1개월 × 3번 호출

### Q4. 첨부파일명이 한글로 깨집니다.
A. 파일 다운로드 시 인코딩 문제일 수 있습니다:
1. UTF-8 인코딩으로 파일명 처리
2. URL 디코딩 적용
3. 운영체제별 파일명 규칙 확인