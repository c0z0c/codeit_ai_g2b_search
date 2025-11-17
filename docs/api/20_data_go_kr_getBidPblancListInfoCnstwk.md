---
layout: default
title: "데이타포탈 - 입찰공고목록 정보에 대한 공사조회 API"
description: "데이타포탈 - 입찰공고목록 정보에 대한 공사조회 API"
date: 2025-11-17
author: "김명환"
cache-control: no-cache
expires: 0
pragma: no-cache
---

# 입찰공고목록 정보에 대한 공사조회 API

## 1. API 개요

### 1.1. 서비스 정보
- **API명**: getBidPblancListInfoCnstwk
- **서비스**: 나라장터 입찰공고정보서비스 (BidPublicInfoService)
- **용도**: 공사 분야 입찰공고 목록 조회

### 1.2. 설명
검색조건에 등록일시, 입찰공고번호, 변경일시를 입력하여 나라장터의 입찰공고번호, 공고명, 발주기관, 수요기관, 계약체결방법명 등 공사부분의 입찰공고 정보를 조회합니다.

### 1.3. 성능 정보
- **최대 메시지 사이즈**: 4000 bytes
- **평균 응답 시간**: 500 ms
- **초당 최대 트랜잭션**: 30 tps

## 2. 엔드포인트

### 2.1. Base URL
```
https://apis.data.go.kr/1230000/ad/BidPublicInfoService/getBidPblancListInfoCnstwk

https://apis.data.go.kr/1230000/ad/BidPublicInfoService/getBidPblancListInfoCnstwk?serviceKey=8fea481d8c22a53e6baa58ac84350a16577b68d4ba8ee4b2c5ce3057db4bbbfa&pageNo=1&numOfRows=10&inqryDiv=1&inqryBgnDt=202511150000&type=json&inqryEndDt=202511170000&bidNtceNo=20200100008
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
| inqryDiv | 조회구분 | String(1) | 필수 | 1: 등록일시<br>2: 입찰공고번호<br>3: 변경일시 |
| inqryBgnDt | 조회시작일시 | String(12) | 조건부 필수 | YYYYMMDDHHMM 형식<br>(inqryDiv가 1 또는 3일 때 필수) |
| inqryEndDt | 조회종료일시 | String(12) | 조건부 필수 | YYYYMMDDHHMM 형식<br>(inqryDiv가 1 또는 3일 때 필수) |
| bidNtceNo | 입찰공고번호 | String(40) | 조건부 필수 | 입찰공고번호<br>(inqryDiv가 2일 때 필수) |

### 3.3. 요청 예시

#### 등록일시 기준 조회
```
https://apis.data.go.kr/1230000/ad/BidPublicInfoService/getBidPblancListInfoCnstwk?serviceKey=8fea481d8c22a53e6baa58ac84350a16577b68d4ba8ee4b2c5ce3057db4bbbfa&pageNo=1&numOfRows=10&inqryDiv=1&type=json&inqryBgnDt=202511150000&inqryEndDt=202511170000
```

#### 입찰공고번호 기준 조회
```
https://apis.data.go.kr/1230000/ad/BidPublicInfoService/getBidPblancListInfoCnstwk?serviceKey=8fea481d8c22a53e6baa58ac84350a16577b68d4ba8ee4b2c5ce3057db4bbbfa&pageNo=1&numOfRows=10&inqryDiv=2&type=json&bidNtceNo=R25BK01156411
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
        { 입찰공고 상세 정보 객체 }
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
| items | 입찰공고 목록 | Array | 입찰공고 정보 배열 |

### 4.4. Items 필드 (입찰공고 상세 정보)

#### 4.4.1. 기본 정보

| 필드명 | 한글명 | 타입 | 설명 |
|--------|--------|------|------|
| bidNtceNo | 입찰공고번호 | String(40) | 차세대 나라장터 번호체계:<br>R+년도(2)+단계구분(2)+순번(8)<br>- BK: 입찰<br>- TA: 계약<br>- DD: 발주계획<br>- BD: 사전규격 |
| bidNtceOrd | 입찰공고차수 | String(3) | 재공고/재입찰 시 증가하는 순번 |
| reNtceYn | 재공고여부 | String(1) | Y: 재공고<br>N: 최초공고 |
| rgstTyNm | 등록유형명 | String(100) | "조달청 또는 나라장터 자체 공고건"<br>"나라장터 기타 공고건" |
| ntceKindNm | 공고종류명 | String(100) | 등록공고, 변경공고, 취소공고, 재공고 |

#### 4.4.2. 공고 정보

| 필드명 | 한글명 | 타입 | 설명 |
|--------|--------|------|------|
| bidNtceDt | 입찰공고일시 | String(19) | YYYY-MM-DD HH:MM:SS 형식 |
| refNo | 참조번호 | String(105) | 자체 전자조달시스템 공고번호 또는 G2B 번호 |
| bidNtceNm | 입찰공고명 | String(1000) | 공사명 또는 사업명 |
| intrbidYn | 국제입찰여부 | String(1) | Y: 국제입찰 대상<br>N: 국내입찰 |

#### 4.4.3. 기관 정보

| 필드명 | 한글명 | 타입 | 설명 |
|--------|--------|------|------|
| ntceInsttCd | 공고기관코드 | String(7) | 행자부 코드 또는 조달청 부여 코드 |
| ntceInsttNm | 공고기관명 | String(400) | 공고를 등록하는 기관명 |
| dminsttCd | 수요기관코드 | String(7) | 실제 수요가 발생한 기관 코드 |
| dminsttNm | 수요기관명 | String(400) | 실제 수요가 발생한 기관명 |
| ntceInsttOfclNm | 공고기관담당자명 | String(40) | 공고기관 담당자 성명 |
| ntceInsttOfclTelNo | 공고기관담당자전화번호 | String(20) | 공고기관 담당자 전화번호 |
| ntceInsttOfclEmailAdrs | 공고기관담당자이메일주소 | String(60) | 공고기관 담당자 이메일 |

#### 4.4.4. 입찰 방법 및 일정

| 필드명 | 한글명 | 타입 | 설명 |
|--------|--------|------|------|
| bidMethdNm | 입찰방법명 | String(100) | 전자입찰, 전자시담 등 |
| cntrctCnclsMthdNm | 계약체결방법명 | String(100) | 일반경쟁, 수의계약 등 |
| bidBeginDt | 입찰시작일시 | String(19) | YYYY-MM-DD HH:MM:SS 형식 |
| bidClseDt | 입찰마감일시 | String(19) | YYYY-MM-DD HH:MM:SS 형식 |
| opengDt | 개찰일시 | String(19) | YYYY-MM-DD HH:MM:SS 형식 |
| bidQlfctRgstDt | 입찰참가자격등록마감일시 | String(16) | YYYY-MM-DD HH:MM 형식 |
| rbidPermsnYn | 재입찰허용여부 | String(1) | Y/N |
| rbidOpengDt | 재입찰개찰일시 | String(19) | YYYY-MM-DD HH:MM:SS 형식 |

#### 4.4.5. 금액 정보

| 필드명 | 한글명 | 타입 | 설명 |
|--------|--------|------|------|
| bdgtAmt | 예산금액 | String(15) | 예산금액 (단위: 원) |
| presmptPrce | 추정가격 | String(15) | 추정가격 (단위: 원) |
| govsplyAmt | 관급자재금액 | String(15) | 관급자재 금액 (단위: 원) |
| contrctrcnstrtnGovsplyMtrlAmt | 계약자부담관급자재금액 | String(15) | 계약자 부담 관급자재 금액 |
| govcnstrtnGovsplyMtrlAmt | 발주기관부담관급자재금액 | String(15) | 발주기관 부담 관급자재 금액 |
| VAT | 부가세 | String(15) | 부가가치세 (단위: 원) |

#### 4.4.6. 예가 정보

| 필드명 | 한글명 | 타입 | 설명 |
|--------|--------|------|------|
| prearngPrceDcsnMthdNm | 예정가격결정방법명 | String(100) | 복수예가, 단일예가 등 |
| totPrdprcNum | 총예정가격수 | String(2) | 예정가격 생성 개수 |
| drwtPrdprcNum | 추첨예정가격수 | String(2) | 추첨할 예정가격 개수 |
| rsrvtnPrceReMkngMthdNm | 예비가격재작성방법명 | String(400) | 재입찰 시 예비가격 재작성 방법 |

#### 4.4.7. 낙찰 정보

| 필드명 | 한글명 | 타입 | 설명 |
|--------|--------|------|------|
| sucsfbidMthdCd | 낙찰방법코드 | String(9) | 한글(1자리) + 숫자(6자리) |
| sucsfbidMthdNm | 낙찰방법명 | String(500) | 낙찰자 결정 방법 |
| sucsfbidLwltRate | 낙찰하한율 | String(5) | 낙찰 하한율 (%) |

#### 4.4.8. 공동수급 정보

| 필드명 | 한글명 | 타입 | 설명 |
|--------|--------|------|------|
| cmmnSpldmdMethdCd | 공동수급방식코드 | String(15) | 한글(1자리) + 숫자(6자리) |
| cmmnSpldmdMethdNm | 공동수급방식명 | String(200) | 공동수급 허용/불허 방식 |
| cmmnSpldmdAgrmntRcptdocMethd | 공동수급협정서접수방법 | String(200) | 협정서 접수 방법 |
| cmmnSpldmdAgrmntClseDt | 공동수급협정서마감일시 | String(16) | YYYY-MM-DD HH:MM 형식 |
| cmmnSpldmdCorpRgnLmtYn | 공동수급지역제한여부 | String(1) | Y/N |
| cmmnSpldmdCnum | 공동수급업체수 | String(200) | 공동수급 가능 업체 수 |

#### 4.4.9. 지역/업종 제한

| 필드명 | 한글명 | 타입 | 설명 |
|--------|--------|------|------|
| rgnDutyJntcontrctYn | 지역의무공동도급여부 | String(1) | Y/N |
| jntcontrctDutyRgnNm1~3 | 공동도급의무지역명1~3 | String(400) | 공동도급 의무 지역 |
| rgnDutyJntcontrctRt | 지역의무공동도급비율 | String(7) | 의무 비율 (%) |
| indstrytyLmtYn | 업종제한여부 | String(1) | Y/N |
| indstrytyEvlRt | 업종평가비율 | String(4) | 업종 평가 비율 (%) |
| rgnLmtBidLocplcJdgmBssCd | 지역제한입찰지역판단기준코드 | String(1) | 지역 판단 기준 |
| rgnLmtBidLocplcJdgmBssNm | 지역제한입찰지역판단기준명 | String(100) | 지역 판단 기준명 |
| cnstrtsiteRgnNm | 공사현장지역명 | String(400) | 공사 현장 위치 |

#### 4.4.10. 공사 정보

| 필드명 | 한글명 | 타입 | 설명 |
|--------|--------|------|------|
| mainCnsttyNm | 주공종명 | String(200) | 주요 공종 명칭 |
| mainCnsttyCnstwkPrearngAmt | 주공종공사예정금액 | String(15) | 주공종 예정금액 (단위: 원) |
| mainCnsttyPresmptPrce | 주공종추정가격 | String(15) | 주공종 추정가격 (단위: 원) |
| subsiCnsttyNm1~9 | 보조공종명1~9 | String(200) | 보조 공종 명칭 |
| subsiCnsttyIndstrytyEvlRt1~9 | 보조공종업종평가비율1~9 | String(4) | 보조공종 평가비율 (%) |
| cnsttyAccotShreRateList | 공종별분담비율목록 | String(200) | 공종별 분담 비율 |

#### 4.4.11. 입찰 제한 및 참가

| 필드명 | 한글명 | 타입 | 설명 |
|--------|--------|------|------|
| dtlsBidYn | 세부입찰여부 | String(1) | Y/N |
| bidPrtcptLmtYn | 입찰참가제한여부 | String(1) | Y/N |
| bidPrtcptFeePaymntYn | 입찰참가수수료납부여부 | String(1) | Y/N |
| bidPrtcptFee | 입찰참가수수료 | String(15) | 참가 수수료 금액 |
| bidGrntymnyPaymntYn | 입찰보증금납부여부 | String(1) | Y/N |
| bidWgrnteeRcptClseDt | 입찰보증금접수마감일시 | String(16) | YYYY-MM-DD HH:MM 형식 |
| ciblAplYn | 건설산업기본법적용여부 | String(1) | Y/N |

#### 4.4.12. 평가 관련

| 필드명 | 한글명 | 타입 | 설명 |
|--------|--------|------|------|
| pqEvalYn | 사전심사평가여부 | String(1) | Y/N |
| pqApplDocRcptMthdNm | 사전심사신청서접수방법명 | String(100) | 사전심사 접수 방법 |
| pqApplDocRcptDt | 사전심사신청서접수일시 | String(16) | YYYY-MM-DD HH:MM 형식 |
| dsgntCmptYn | 지정경쟁여부 | String(1) | Y/N |
| arsltCmptYn | 실적경쟁여부 | String(1) | Y/N |
| arsltApplDocRcptMthdNm | 실적신청서접수방법명 | String(100) | 실적신청서 접수 방법 |
| arsltApplDocRcptDt | 실적신청서접수일시 | String(16) | YYYY-MM-DD HH:MM 형식 |
| cnstrtnAbltyEvlAmtList | 시공능력평가금액목록 | String(200) | 시공능력 평가 금액 |
| indstrytyMfrcFldEvlYn | 업종제조분야평가여부 | String(1) | Y/N |

#### 4.4.13. 인센티브 지역

| 필드명 | 한글명 | 타입 | 설명 |
|--------|--------|------|------|
| incntvRgnNm1~4 | 인센티브지역명1~4 | String(400) | 인센티브 적용 지역 |

#### 4.4.14. 기타 정보

| 필드명 | 한글명 | 타입 | 설명 |
|--------|--------|------|------|
| aplBssCntnts | 적용기준내용 | String(500) | 적용 기준 (예: 행자부) |
| opengPlce | 개찰장소 | String(200) | 개찰 장소 |
| dcmtgOprtnDt | 현장설명일시 | String(16) | YYYY-MM-DD HH:MM 형식 |
| dcmtgOprtnPlce | 현장설명장소 | String(200) | 현장설명회 장소 |
| crdtrNm | 채권자명 | String(100) | 계약 채권자 |
| exctvNm | 집행담당자명 | String(40) | 집행 담당자 |
| rgstDt | 등록일시 | String(19) | YYYY-MM-DD HH:MM:SS 형식 |
| chgDt | 변경일시 | String(19) | YYYY-MM-DD HH:MM:SS 형식 |
| chgNtceRsn | 변경공고사유 | String(2000) | 변경 공고 사유 |
| bfSpecRgstNo | 사전규격등록번호 | String(40) | 사전규격 등록번호 |
| orderPlanUntyNo | 발주계획통합번호 | String(40) | 발주계획 통합 번호 |
| untyNtceNo | 통합공고번호 | String(40) | 통합 공고 번호 |
| brffcBidprcPermsnYn | 부정당업체입찰허용여부 | String(1) | Y/N |
| mtltyAdvcPsblYn | 복수업체낙찰가능여부 | String(1) | Y/N |
| mtltyAdvcPsblYnCnstwkNm | 복수업체낙찰가능여부공사명 | String(200) | 복수낙찰 대상 공사 |
| ntceDscrptYn | 공고설명여부 | String(1) | Y/N |
| dminsttOfclEmailAdrs | 수요기관담당자이메일주소 | String(60) | 수요기관 담당자 이메일 |
| indutyVAT | 산업단지부가세 | String(15) | 산업단지 부가세 |

#### 4.4.15. 첨부파일 URL

| 필드명 | 한글명 | 타입 | 설명 |
|--------|--------|------|------|
| ntceSpecDocUrl1~10 | 공고상세문서URL1~10 | String(800) | 공고 첨부파일 다운로드 URL |
| ntceSpecFileNm1~10 | 공고상세파일명1~10 | String(200) | 공고 첨부파일명 |
| sptDscrptDocUrl1~5 | 보충설명문서URL1~5 | String(800) | 보충설명 문서 URL |
| stdNtceDocUrl | 표준공고문서URL | String(800) | 표준 공고문서 URL |

#### 4.4.16. 상세 페이지 URL

| 필드명 | 한글명 | 타입 | 설명 |
|--------|--------|------|------|
| bidNtceDtlUrl | 입찰공고상세URL | String(800) | 나라장터 입찰공고 상세 페이지 URL |
| bidNtceUrl | 입찰공고URL | String(800) | 입찰공고 URL |

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
          "bidNtceNo": "R25BK01156411",
          "bidNtceOrd": "000",
          "reNtceYn": "N",
          "rgstTyNm": "조달청 또는 나라장터 자체 공고건",
          "ntceKindNm": "등록공고",
          "intrbidYn": "N",
          "bidNtceDt": "2025-11-15 16:49:41",
          "refNo": "산림산업과-27463",
          "bidNtceNm": "제암산자연휴양림 더늠길(곰재화장실 구간) 보완사업",
          "ntceInsttCd": "4890000",
          "ntceInsttNm": "전라남도 보성군",
          "dminsttCd": "4890000",
          "dminsttNm": "전라남도 보성군",
          "bidMethdNm": "전자입찰",
          "cntrctCnclsMthdNm": "수의계약",
          "ntceInsttOfclNm": "박성래",
          "ntceInsttOfclTelNo": "061-850-5161",
          "ntceInsttOfclEmailAdrs": "sparkc@korea.kr",
          "exctvNm": "박성래",
          "bidQlfctRgstDt": "2025-11-20 18:00",
          "bidBeginDt": "2025-11-19 10:00:00",
          "bidClseDt": "2025-11-21 10:00:00",
          "opengDt": "2025-11-21 11:00:00",
          "rbidPermsnYn": "Y",
          "rbidOpengDt": "2025-11-21 11:00:00",
          "prearngPrceDcsnMthdNm": "복수예가",
          "totPrdprcNum": "15",
          "drwtPrdprcNum": "4",
          "bdgtAmt": "222882000",
          "presmptPrce": "127594546",
          "govsplyAmt": "0",
          "contrctrcnstrtnGovsplyMtrlAmt": "82528000",
          "govcnstrtnGovsplyMtrlAmt": "0",
          "VAT": "12759454",
          "sucsfbidMthdCd": "낙030029",
          "sucsfbidMthdNm": "소액수의견적(2인 이상 견적 제출)-국민연금보험료 등 합산액 감액 적용",
          "sucsfbidLwltRate": "89.745",
          "aplBssCntnts": "행자부",
          "mainCnsttyNm": "산림사업법인(자연휴양림등 조성)",
          "mainCnsttyCnstwkPrearngAmt": "222882000",
          "indstrytyLmtYn": "Y",
          "cnstrtsiteRgnNm": "전라남도 보성군",
          "rgnDutyJntcontrctYn": "N",
          "cmmnSpldmdMethdNm": "(없음)공동수급불허",
          "opengPlce": "국가종합전자조달시스템(나라장터)",
          "crdtrNm": "보성군수",
          "cmmnSpldmdCnum": "공고서에 의함",
          "untyNtceNo": "R25BM00480773",
          "rsrvtnPrceReMkngMthdNm": "재입찰시 예비가격을 다시 생성하여 예정가격이 산정됩니다.",
          "rgstDt": "2025-11-15 16:49:41",
          "dsgntCmptYn": "N",
          "arsltCmptYn": "N",
          "ciblAplYn": "N",
          "mtltyAdvcPsblYn": "N",
          "indstrytyMfrcFldEvlYn": "N",
          "rgnLmtBidLocplcJdgmBssCd": "N",
          "rgnLmtBidLocplcJdgmBssNm": "본사또는참여지사소재지",
          "ntceSpecDocUrl1": "https://www.g2b.go.kr/pn/pnp/pnpe/UntyAtchFile/downloadFile.do?bidPbancNo=R25BK01156411&bidPbancOrd=000&fileType=&fileSeq=1&prcmBsneSeCd=07",
          "ntceSpecFileNm1": "공고문(제암산자연휴양림 더늠길 데크 보완사업).hwp",
          "stdNtceDocUrl": "https://www.g2b.go.kr/pn/pnp/pnpe/UntyAtchFile/downloadFile.do?bidPbancNo=R25BK01156411&bidPbancOrd=000&fileType=&fileSeq=1&prcmBsneSeCd=07",
          "bidNtceDtlUrl": "https://www.g2b.go.kr/link/PNPE027_01/single/?bidPbancNo=R25BK01156411&bidPbancOrd=000"
        }
      ],
      "numOfRows": 10,
      "pageNo": 1,
      "totalCount": 9
    }
  }
}
```

### 5.2. 오류 응답

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

### 7.1. 데이터 조회 제한
- 조회구분(inqryDiv)에 따라 필수 파라미터가 달라집니다.
- 등록일시/변경일시 조회 시: inqryBgnDt, inqryEndDt 필수
- 입찰공고번호 조회 시: bidNtceNo 필수

### 7.2. 공고번호 체계
- **차세대 나라장터 번호**: R+년도(2)+단계구분(2)+순번(8) = 총 13자리
- 단계구분 코드:
  - BK: 입찰
  - TA: 계약
  - DD: 발주계획
  - BD: 사전규격

### 7.3. 응답 데이터 특성
- 빈 값인 필드는 빈 문자열("")로 반환됩니다.
- 옵션 필드는 조건에 따라 값이 없을 수 있습니다.
- URL 필드는 실제 파일 다운로드 링크로 제공됩니다.

### 7.4. 성능 고려사항
- 한 번에 대량의 데이터 조회 시 pageNo와 numOfRows를 적절히 활용하세요.
- 불필요한 반복 호출을 피하고 적절한 캐싱 전략을 수립하세요.

## 8. 활용 예시

### 8.1. Python 예시

```python
import requests
import json

def get_bid_construction_info(service_key, inqry_div, **kwargs):
    """
    공사 입찰공고 조회
    
    Args:
        service_key (str): 공공데이터포탈 인증키
        inqry_div (str): 조회구분 (1: 등록일시, 2: 입찰공고번호, 3: 변경일시)
        **kwargs: 추가 파라미터
    
    Returns:
        dict: API 응답 데이터
    """
    url = "https://apis.data.go.kr/1230000/ad/BidPublicInfoService/getBidPblancListInfoCnstwk"
    
    params = {
        "serviceKey": service_key,
        "pageNo": kwargs.get("pageNo", 1),
        "numOfRows": kwargs.get("numOfRows", 10),
        "type": "json",
        "inqryDiv": inqry_div
    }
    
    # 조회구분에 따른 필수 파라미터 추가
    if inqry_div in ["1", "3"]:
        params["inqryBgnDt"] = kwargs.get("inqryBgnDt")
        params["inqryEndDt"] = kwargs.get("inqryEndDt")
    elif inqry_div == "2":
        params["bidNtceNo"] = kwargs.get("bidNtceNo")
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API 호출 실패: {response.status_code}")

# 사용 예시
service_key = "your_service_key_here"

# 등록일시 기준 조회
result = get_bid_construction_info(
    service_key=service_key,
    inqry_div="1",
    inqryBgnDt="202511150000",
    inqryEndDt="202511170000"
)

# 결과 출력
if result["response"]["header"]["resultCode"] == "00":
    items = result["response"]["body"]["items"]
    for item in items:
        print(f"공고번호: {item['bidNtceNo']}")
        print(f"공고명: {item['bidNtceNm']}")
        print(f"공고기관: {item['ntceInsttNm']}")
        print(f"추정가격: {item['presmptPrce']}원")
        print("-" * 50)
else:
    print(f"오류: {result['response']['header']['resultMsg']}")
```

### 8.2. JavaScript 예시

```javascript
async function getBidConstructionInfo(serviceKey, inqryDiv, options = {}) {
    const baseUrl = 'https://apis.data.go.kr/1230000/ad/BidPublicInfoService/getBidPblancListInfoCnstwk';
    
    const params = new URLSearchParams({
        serviceKey: serviceKey,
        pageNo: options.pageNo || 1,
        numOfRows: options.numOfRows || 10,
        type: 'json',
        inqryDiv: inqryDiv
    });
    
    // 조회구분에 따른 필수 파라미터 추가
    if (['1', '3'].includes(inqryDiv)) {
        params.append('inqryBgnDt', options.inqryBgnDt);
        params.append('inqryEndDt', options.inqryEndDt);
    } else if (inqryDiv === '2') {
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

// 사용 예시
const serviceKey = 'your_service_key_here';

// 등록일시 기준 조회
getBidConstructionInfo(serviceKey, '1', {
    inqryBgnDt: '202511150000',
    inqryEndDt: '202511170000'
}).then(result => {
    if (result.response.header.resultCode === '00') {
        const items = result.response.body.items;
        items.forEach(item => {
            console.log(`공고번호: ${item.bidNtceNo}`);
            console.log(`공고명: ${item.bidNtceNm}`);
            console.log(`공고기관: ${item.ntceInsttNm}`);
            console.log(`추정가격: ${item.presmptPrce}원`);
            console.log('-'.repeat(50));
        });
    } else {
        console.error(`오류: ${result.response.header.resultMsg}`);
    }
});
```

## 9. 참고사항

### 9.1. 관련 API
- getBidPblancListInfoServc: 용역 입찰공고 조회
- getBidPblancListInfoFrgcpt: 외자 입찰공고 조회
- getBidPblancListInfoThng: 물품 입찰공고 조회
- getBidPblancListInfoCnstwkBsisAmount: 공사 기초금액 조회
- getBidPblancListInfoChgHstryCnstwk: 공사 변경이력 조회

### 9.2. 데이터 갱신 주기
- 입찰공고 등록 시 실시간 반영
- 변경사항 발생 시 즉시 업데이트

### 9.3. 문의처
- 공공데이터포탈: www.data.go.kr
- 나라장터: www.g2b.go.kr
- 조달청 고객센터: 1588-0800