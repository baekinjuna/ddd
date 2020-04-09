import itertools
import numpy as np
import logging
import json
from django.db.models import Q
from django.utils import timezone
from django.core.paginator import Paginator
from collections import Counter
from rest_framework import status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from core.models import SMK, Manager, SearchGroup, docu2vec, Trajectory, Search_Word_Path, Document, Info_Document, Userid, Consumed_contents, which_reco
from .serializers import SearchResultSerializer
import json, os
from django.conf import settings
from scipy.stats import beta
from datetime import datetime
import library.lib_konlpy as lib_konlpy
import codecs
from bs4 import BeautifulSoup
from gensim.models.doc2vec import Doc2Vec
import library.knn as knn
from operator import itemgetter
from konlpy.tag import Mecab
from itertools import chain, groupby
import datetime as dt
import collections
import time
import numpy as np
from django.db import connection

log = logging.getLogger(__name__)


class GeneralResponse:
    def __init__(self, status_code, message, payload):
        self.response = {
            'status_code': status_code,
            'message': message,
            'payload': payload if payload else {}
        }


class InitInfo(APIView):  # Info_Document Model을 모두 initializing 하는 api
    permission_classes = (AllowAny,)

    def get(self, request):
        objs = Info_Document.objects.all()
        for obj in objs:
            obj.clicked = 1
            obj.recommended = 1
            obj.score = 0.5
            obj.save()
        return Response(
            status=status.HTTP_200_OK,
            data=GeneralResponse(
                200,
                'InitInfo, get',
                {}
            ).response
        )


class RecentWord(APIView):  # 최근 검색어를 받아 오는 것
    permission_classes = (AllowAny,)

    def get(self, request):
        session = request.GET.get('session_id')  # 세션 아이디를 받는다
        check = Userid.objects.filter(session=session)  # 해당하는 세션이 DB내에 존재하는지 확인한다
        if check:
            objs = Search_Word_Path.objects.filter(session=session)  # 해당하는 세션이 DB내에 존재한다면 최근 검색어를 모두 가져온다
            word_list = [obj.search_word for obj in objs]  # 그 중 search_word만 list로 만든다
            word_list = groupby(word_list)  # 연속되는 검색어를 하나로 줄인다
            x = [k for k, v in word_list]
            x.reverse()
            x = x[:5]  # 최근 5개만 받는다
            return Response(
                status=status.HTTP_200_OK,
                data=GeneralResponse(
                    200,
                    'RecentWord, get',
                    x
                ).response
            )
        else:
            return Response(
                status=status.HTTP_400_BAD_REQUEST,
                data=GeneralResponse(
                    400,
                    'session error, RecentWord, get',
                    []
                ).response
            )


class FamousSearch(APIView):  # 인기 검색어를 받는 것
    permission_classes = (AllowAny,)

    def get(self, request):
        today = dt.date.today()  # 오늘 날짜 구한다
        tomorrow = today + dt.timedelta(days=1)
        yesterday = today - dt.timedelta(days=7)  # 일주일 전 날짜 구한다
        objs = Search_Word_Path.objects.filter(search_date__range=[yesterday, tomorrow])  # 일주일 내 검색이력을 들고 온다
        x = [x.search_word for x in objs]  # 검색어를 list로 만든다
        x = collections.Counter(x)  # Counter한다.(dict형태)
        x = [(l, k) for k, l in sorted([(j, i) for i, j in x.items()], reverse=True)]  # dict형태를 value로 sorting한다
        x = [l[0] for l in x][:10]  # 제일 많은 10개를 들고온다
        return Response(
            status=status.HTTP_200_OK,
            data=GeneralResponse(
                200,
                'FamousSearch, get',
                x
            ).response
        )


class RecentDocument(APIView):  # 최근 읽은 컨텐츠 이력 가져오기
    permission_classes = (AllowAny,)

    def get(self, request):
        session = request.GET.get('session_id')  # 세션을 받는다
        obj = Userid.objects.get(session=session)  # 세션이 존재하는지 확인하고 object를 들고온다
        d = []
        already = json.loads(obj.already)[-20:]  # 읽은 컨텐츠 최근 20개 가져오기
        already = groupby(already)  # 연속되는 것 하나로 만들기
        already = [k for k, v in already]
        for i in range(len(already)):  # Response 형태 만들기
            obj = Document.objects.get(application_number=already[i])
            d.append({'key': i, 'title': obj.title_korean, 'application_number': obj.application_number})
        d.reverse()
        d = d[:5]  # 최근 5개만 보내주기
        return Response(
            status=status.HTTP_200_OK,
            data=GeneralResponse(
                200,
                'RecentDocument, get',
                d
            ).response
        )


class OneDocument(APIView):  # 컨텐츠 하나에 대한 자세한 정보 가져오기
    permission_classes = (AllowAny,)

    def get(self, request, pk):
        obj = Document.objects.get(application_number=pk)  # 특허고유값(출원번호)를 받은 후 Object를 가져온다
        family = []
        doc_family = []

        xa = (obj.family_country_name.split('|')[:-1])  # Family 국가이름 List
        xb = (obj.family_literature_kind.split('|')[:-1])  # Family 종류 List
        xc = (obj.family_number.split('|')[:-1])  # Family 숫자 List
        xd = (obj.family_kind.split('|')[:-1])  # Family kind List
        doc = 0
        fam = 0

        d = {}
        if obj.new_search_group == None:  # 연구단이 입력되지 않았을 경우
            iix = ''
            empty = []

        else:  # 연구단이 입력되었을 경우
            group_list = json.loads(obj.new_search_group)  # json.load로 리스트 풀기
            xxs = SearchGroup.objects.filter(id__in=group_list)  # 푼 리스트를 이용하여 SearchGroup Model에서 object 가져오기
            manager_id = []
            for xx in xxs:  # Object Query를 For문 돌기
                a = json.loads(xx.manager)  # manager 값 가져오기
                manager_id_list = Manager.objects.filter(id__in=a)  # manager 값으로 Manager object 가져오기
                mana = [x.id for x in manager_id_list]  # Manager id 가져오기
                manager_id.append(mana)  # manager_id라는 리스트에 하나씩 붙여넣기
            group = [x.name for x in xxs]  # group 리스트에 연구단 이름 붙여넣기
            iix = ''
            for ix in group:  # group의 리스트를 ','로 이어준다
                iix += ix + ','
            d['new_search_group'] = iix[:-1]  # 제일 마지막에 ','가 붙는데 이를 제거한다
            manager_id = sum(manager_id, [])  # manager_id의 리스트를 flatten한다.
            asdf = []
            for iff in manager_id:  # manager_id 리스트 안에 중복되는 값들이 있을 수 있기에 중복되지 않는 것만 선택하여 다시 리스트(asdf)로 만든다.
                if not iff in asdf:
                    asdf.append(iff)
            manager_id = asdf  # 임시변수(asdf)를 manager_id로 대체한다
            empty = []
            for manager in manager_id:  # manager_id를 이용하여 dict형태의 이름, 전화번호, 이메일을 만듬.
                mana = Manager.objects.get(id=manager)
                ddd = {'name': mana.name, 'phone': mana.phone, 'mail': mana.mail}
                empty.append(ddd)
            d['manager'] = empty

        for a, b, c, d in zip(xa, xb, xc, xd):  # Family 정보를 이용하여 doc_family와 family로 나누어 저장
            if 'DOCDB' in d:
                doc_family.append({'key': doc, 'nation_name': a, 'category': b, 'nation_code': c[:2], 'family_number': c[2:], 'doc_flag': d})
                doc += 1
            else:
                family.append({'key': fam, 'nation_name': a, 'category': b, 'nation_code': c[:2], 'family_number': c[2:], 'doc_flag': d})
                fam += 1

        # Check pdf exist
        application_pdf_path = '{}pdf_directory/{}.pdf'.format(
            settings.MEDIA_ROOT, obj.application_number
        )
        registration_pdf_path = '{}pdf_registration_directory/{}.pdf'.format(
            settings.MEDIA_ROOT, obj.application_number
        )
        application_pdf_exist = False
        registration_pdf_exist = False
        if os.path.isfile(application_pdf_path):
            application_pdf_exist = True
        if os.path.isfile(registration_pdf_path):
            registration_pdf_exist = True

        data = {
            'title': {
                'title_korean': obj.title_korean,
                'title_english': obj.title_english
            },
            'tech_info': {
                'document_id': obj.id,
                'applicant': obj.applicant.replace('|', ',')[:-1],
                'application_number': obj.application_number,
                'application_date': obj.application_date,
                'registration_number': obj.registration_number,
                'registration_date': obj.registration_date,
                'release_number': obj.release_number,
                'release_date': obj.release_date,
                'tech_transfer_charge': {
                    'tech_transfer_name': obj.tech_transfer_name,
                    'tech_transfer_phone': obj.tech_transfer_phone,
                    'tech_transfer_mail': obj.tech_transfer_mail,
                    'manager': empty
                },
                'search_group': iix[:-1],
                'summary': obj.summary,
                'legal_status': obj.legal_status,
                'judging_progress_status': obj.judging_progress_status
            },
            'claim': obj.claim.split('|')[:-1],
            'family': {
                'family': family,
                'doc_family': doc_family
            },
            'application_pdf_exist': application_pdf_exist,
            'registration_pdf_exist': registration_pdf_exist
        }
        # data = obj.family_country_name.split('|')
        return Response(
            status=status.HTTP_200_OK,
            data=GeneralResponse(
                200,
                'OneDocument',
                data
            ).response
        )


class MainPageContent(APIView):  # 사용자 페이지 첫 화면에서 Content를 불러오는 API
    permission_classes = (AllowAny,)

    def get(self, request):
        # Added for paging
        page = request.GET.get('page', 1)
        page_count = request.GET.get('page_count', 10)
        # button: 메인페이지 버튼 (1 = 최근성과, 2 = 등록특허, 3 = 출원건수)
        button = request.GET.get('button', 1)
        # is_all: 전체 페이지 호출 여부 ('true', 'false')
        sort = request.GET.get('sort')
        sort_type = request.GET.get('sort_type')
        is_all = request.GET.get('is_all', 'false')
        if is_all == 'true':
            is_all = True
        else:
            is_all = False
        category = request.GET.get('category')  # 카테고리 (문서 참조)
        search = request.GET.get('search')  # 검색어

        if not sort:
            sort = 'application_number'
        if not sort_type :
            sort_type = 'ascend'

        if button == '1':
            objs = Document.objects.all().order_by('-application_date')[:5]  # application-date로 Document 모델을 정렬하여 제일 최근 5개를 가져옴
        elif button == '2':
            objs = Document.objects.filter(legal_status='등록')
        elif button == '3':
            objs = Document.objects

        if search and category:
            object_list = []
            if category == 'application_number':  # 만약 카테고리가 application_number일 경우
                objs = objs.filter(application_number__contains=search)
            elif category == 'title':
                objs = objs.filter(title_korean__icontains=search)
            elif category == 'applicant':
                objs = objs.filter(applicant__icontains=search)
            elif category == 'research_group':
                research_group_instance = SearchGroup.objects.filter(name__contains=search)
                target_documents = []
                for item in research_group_instance:
                    print(str(item.id))
                    target_documents.append(objs.filter(new_search_group__contains=str(item.id)).all())
                filtered_documents = []
                for target in target_documents:
                    for document in target:
                        filtered_documents.append(document.id)
                objs = objs.filter(id__in=filtered_documents)
            elif category == 'application_date':
                # YYYY-MM-DD
                objs = objs.filter(application_date__contains=search)
            elif category == 'registration_date':
                # YYYY-MM-DD
                objs = objs.filter(registration_date__contains=search)
            elif category == 'all':
                a = objs.filter(application_number__contains=search)
                b = objs.filter(title_korean__icontains=search)
                c = objs.filter(applicant__icontains=search)
                d = objs.filter(application_date__contains=search)
                e = objs.filter(registration_date__contains=search)
                objs = a | b | c | d | e  # 찾은 모든 쿼리를 OR로 축소시킨 후 objs로 할당
            else:
                a = objs.filter(application_number__contains=search)
                b = objs.filter(title_korean__icontains=search)
                c = objs.filter(applicant__icontains=search)
                d = objs.filter(application_date__contains=search)
                e = objs.filter(registration_date__contains=search)
                objs = a | b | c | d | e  # 찾은 모든 쿼리를 OR로 축소시킨 후 objs로 할당

        if is_all:
            instance = objs.all()
            total_page = 0
            total_documents = 0
        else:
            if sort_type == 'ascend':
                p = Paginator(objs.all().order_by(sort), page_count)
                instance = p.page(page)
                total_page = p.num_pages
                total_documents = p.count
            else:
                p = Paginator(objs.all().order_by("-"+sort), page_count)
                instance = p.page(page)
                total_page = p.num_pages
                total_documents = p.count
        
        data = []
        for i, obj in enumerate(instance):  # objs를 for 돌면서 d에 할당시킴
            try:
                d = {}
                d['key'] = i
                d['title_korean'] = obj.title_korean
                d['title_english'] = obj.title_english
                d['applicant'] = obj.applicant.replace('|', ',')[:-1]
                d['application_number'] = obj.application_number
                d['application_date'] = obj.application_date
                d['registration_number'] = obj.registration_number
                d['registration_date'] = obj.registration_date
                d['release_number'] = obj.registration_number
                d['release_date'] = obj.registration_date
                d['inventor'] = obj.inventor
                d['inventor_english'] = obj.inventor_english
                try:
                    group_list = json.loads(obj.new_search_group)
                    group_names = SearchGroup.objects.filter(id__in=group_list).values_list('name', flat=True)
                    group_names = ','.join(group_names)
                    d['new_search_group'] = group_names
                except Exception as e:
                    group_names = ''
                    log.exception(e)
                smk_objs = SMK.objects.filter(document=obj)
                smk = []
                for smk_obj in smk_objs:
                    # if os.path.isfile(settings.MEDIA_ROOT + str(smk_obj.file)):
                    if smk_obj.type is 10:
                        smk_temp = {
                            'smk_type': smk_obj.type,
                            'smk_url': str(smk_obj.file),
                            'smk_file_name': smk_obj.original_file_name
                        }
                    else:
                        smk_temp = {
                            'smk_type': smk_obj.type,
                            'smk_url': smk_obj.url,
                            'smk_file_name': 'Youtube SMK'
                        }
                    smk.append(smk_temp)
                d['smk'] = smk
                if obj.new_search_group == None:  # object에 new_search_group이 없을 경우(None)
                    d['new_search_group'] = ''
                    d['manager'] = []

                else:  # object에 new_search_group이 있을 경우
                    group_list = json.loads(obj.new_search_group)  # group_list에 new_search_group을 할당
                    xxs = SearchGroup.objects.filter(id__in=group_list)  # xxs에 group_list를 포함하는 SearchGroup object들을 가져옴
                    manager_id = []
                    for xx in xxs:  # xxs object를 for 돌면서
                        a = json.loads(xx.manager)  # manager list를 a에 저장
                        if a:
                            manager_id_list = Manager.objects.filter(id__in=a)  # a의 인덱스를 포함하는 Manager object를 manager_id_list에 할당
                            mana = [x.id for x in manager_id_list]  # 리스트 돌면서 id를 가져옴
                            manager_id.append(mana)  # manager_id에 붙여넣기함
                    group = [x.name for x in xxs]  # group에 name을 붙여넣기함
                    iix = ''
                    for ix in group:
                        iix += ix + ','  # group 리스트를 돌면서 ','로 붙여넣기
                    d['new_search_group'] = iix[:-1]  # 마지막 ',' 지움
                    manager_id = sum(manager_id, [])  # manager_id를 flatten하게 만듬
                    asdf = []
                    for iff in manager_id:
                        if not iff in asdf:
                            asdf.append(iff)  # 중복되는 요소를 없애기
                    manager_id = asdf  # asdf를 manager_id에 다시 할당
                    empty = []
                    for manager in manager_id:  # 데이터 만들어서 넣는 부분
                        mana = Manager.objects.get(id=manager)
                        ddd = {'name': mana.name, 'phone': mana.phone, 'mail': mana.mail}
                        empty.append(ddd)
                    d['manager'] = empty

                d['transfer_name'] = obj.tech_transfer_name
                d['transfer_mail'] = obj.tech_transfer_mail
                d['transfer_phone'] = obj.tech_transfer_phone
                d['search_group'] = obj.search_group

                if obj.tech_transfer_name != None:
                    d['transfer_manager'] = obj.tech_transfer_name + ' ' + obj.tech_transfer_mail + ' ' + obj.tech_transfer_phone
                else:
                    d['transfer_manager'] = None

                if obj.inventor:
                    d['inventor'] = obj.inventor.replace(',', ' ').replace('|', ', ')[:-2]
                else:
                    d['inventor'] = ''

                if obj.inventor_english:
                    d['inventor_english'] = obj.inventor_english.replace(',', ' ').replace('|', ', ')[:-2]
                else:
                    d['inventor_english'] = ''

                data.append(d)
            except Exception as e:
                log.exception(e)
                continue

        result = {
            'data': data,
            'page': page,
            'page_count': page_count,
            'total_page': total_page,
            'total_documents': total_documents
        }

        return Response(
            status=status.HTTP_200_OK,
            data=GeneralResponse(
                200,
                'mainpagecontent',
                result
            ).response
        )


class MainPage(APIView):  # main page의 3개 카드에 띄우는 숫자를 내어주는 것
    permission_classes = (AllowAny,)

    def get(self, request):
        register_objs = Document.objects.filter(legal_status='등록').count()
        objs = Document.objects.all().count()
        result = {'register': register_objs, 'total': objs}
        return Response(
            status=status.HTTP_200_OK,
            data=GeneralResponse(
                200,
                'mainpage',
                result
            ).response
        )


class AutoComplete(APIView):  # 자동완성 기능
    permission_classes = (AllowAny,)

    def get(self, request):
        keyword = request.GET.get('nouns')  # 검색어를 받아옴
        result_list = []
        objs = Search_Word_Path.objects.filter(search_word__icontains=keyword)  # Search_word_path object에서 search_word에 keyword가 포함되어 있는 object 가져옴
        for obj in objs:
            result_list.append(obj.search_word)  # search_word를 result_list에 할당
        result = dict(Counter(result_list))  # result_list를 dict result로 만듬
        tup_list = []
        for res in result.keys():
            tup_list.append((res, result[res]))  # tup_list에 key, value를 할당
        sorted_by_second = sorted(tup_list, key=lambda tup: tup[1], reverse=True)  # tup_list의 value에 맞춰 정렬
        result = []
        for sor in sorted_by_second:
            result.append(sor[0])  # result에 제일 많이 출현하는 순으로 가져온 후 result에 할당
        return Response(
            status=status.HTTP_200_OK,
            data=GeneralResponse(
                200,
                'success',
                result
            ).response
        )


class RelatedQuery(APIView):  # 연관검색어 찾는 API
    permission_classes = (AllowAny,)

    def get(self, request):
        keyword = request.GET.get('nouns')  # 검색어를 입력받는다
        if not keyword:
            return Response(
                status=status.HTTP_200_OK,
                data=GeneralResponse(
                    200,
                    '키워드를 입력해야 합니다',
                    []
                ).response
            )
        mecab = Mecab()  # 검색어를 형태소 분석하고
        vocabs = mecab.nouns(keyword)  # 명사만 추출한다
        ll = []
        model = Doc2Vec.load('model_doc.doc2vec')  # gensim 모델을 연다
        for i in vocabs:  # 추출된 명사들에 연관된 검색어 상위 10개씩을 뽑는다.
            try:
                a = (np.asarray(model.most_similar(i, topn=10)))
                for ii in a:
                    ll.append((ii[0], float(ii[1])))
            except:
                print('xx')
        sorted_ll = sorted(ll, key=lambda tup: tup[1], reverse=True)  # 상위 10개씩 뽑은 것 들 중 연관정도로 정렬한다.
        ll = []
        for i in sorted_ll:  # 정렬된 것들 순서대로 명사만 뽑는다.
            ll.append(i[0])
        ll_model = []
        for vocab in vocabs:
            objs = Trajectory.objects.filter(departure__icontains=vocab)  # 사람들이 검색한것들을 기반으로 연관검색어를 추출하기 위해 departure에 추출된 명사들과 관계있는 모든 object를 가져온다
            for obj in objs:
                if obj.destination != obj.departure:  # 같은 명사로 departure, destination이 구성된것은 제외하고 보여준다
                    ll_model.append(obj.destination)  # ll_model에 할당
        dict_ll_model = dict(Counter(ll_model)).items()  # dict('명사':출현횟수)로 재구성
        sorted_ll_model = sorted(dict_ll_model, key=lambda tup: tup[1], reverse=True)  # 출현횟수에 따라 정렬
        ll_model = []
        for i in sorted_ll_model:
            ll_model.append(i[0])  # 가장 많이 나온 명사들이 제일 앞에 오게 한 후 리스트에 할당

        result = {'db': ll_model, 'model': ll}  # ll_model은 검색어를 저장한 것을 기반, ll은 gensim 모델에 기반하여 result에 저장
        del model
        return Response(
            status=status.HTTP_200_OK,
            data=GeneralResponse(
                200,
                '연관검색 추천 성공',
                result
            ).response
        )


class OnlySearchDocView(APIView):
    permission_classes = (AllowAny,)

    def get(self, request):
        start_time = datetime.now()
        y = time.time()
        nouns = request.GET.get('nouns')  # 검색어 받기
        start_date = request.GET.get('start_date')  # 날짜 시작 받기
        end_date = request.GET.get('end_date')  # 날짜 끝 받기
        category = request.GET.get('category')  # 카테고리가 있으면 받기
        page = request.GET.get('page')  # 몇 페이지에 해당하는지 받기
        page_count = request.GET.get('page_count')  # 한 페이지에 얼마나 보여줄지 받기
        session = request.GET.get('session_id')  # 검색하는 사람의 세션 받기
        interaction = 1 if any('&' in s for s in nouns) else 0
        nouns_list = nouns.split('&')  # 검색어가 &를 포함하고 있으면 분리
        nn = [x.strip() for x in nouns_list]  # blank(띄워쓰기) 삭제
        sort = request.GET.get('sort')
        sort_type = request.GET.get('sort_type')
        search_word = [lib_konlpy.ext_nouns(n) for n in nn]  # 명사만 추출하기 ex) 제조 금속 & 나노 전자 & 유전자 -> [['제조','금속'],['나노','전자'],[유전자]]
        list_nouns = lib_konlpy.ext_nouns(nouns)  # &에 관계없이 명사추출
        print('OnlySearchDocView')

        if not sort:
            sort = 'application_number'
        if not sort_type:
            sort_type = 'ascend'
        if not page:
            page = 1
        if not page_count:
            page_count = 10

        mecab = Mecab()
        if '&' in nouns:
            split_nouns = nouns.split('&')  # 검색어에 & 있으면 분리
        else:
            split_nouns = [nouns]  # 검색어에 & 없으면 형태에 맞게 변형

        nouns = []
        for i in split_nouns:  # 검색어를 &로 나눈 후 형태소 분석 하였을때, 고유명사, 명사, 숫자, 영어가 있는 확인 후 nouns에 저장
            nn = []
            x = mecab.pos(i)
            for a in x:
                if a[1] == 'NNG' or a[1] == 'NNP' or a[1] == 'SN' or a[1] == 'SL':
                    nn.append(a[0])
            nouns.append(nn)
            del x

        bold_nouns = [item for items in nouns for item in items]  # 볼드 처리해야할 단어들

        obj_list = []
        direct_query = []
        for i in nouns:  # nouns를 돌면서 고유명사, 명사, 숫자, 영어를 포함하는 object들을 들고옴
            for num, j in enumerate(i):
                if sort_type == 'ascend':
                    try:
                        application_number = int(j)
                        obj_content = Document.objects.defer('summary', 'claim').filter(
                            Q(content__icontains=j) | Q(title_english__icontains=j) | Q(application_number__icontains=application_number)).order_by(sort)
                    except Exception as e:
                        obj_content = Document.objects.defer('summary', 'claim').filter(Q(content__icontains=j) | Q(title_english__icontains=j)).order_by(sort)
                else:
                    try:
                        application_number = int(j)
                        obj_content = Document.objects.defer('summary', 'claim').filter(
                            Q(content__icontains=j) | Q(title_english__icontains=j) | Q(application_number__icontains=application_number)).order_by('-'+sort)
                    except Exception as e:
                        obj_content = Document.objects.defer('summary', 'claim').filter(Q(content__icontains=j) | Q(title_english__icontains=j)).order_by('-'+sort)
                if num == 0:
                    obj_list = obj_content
                else:
                    obj_list = obj_content

        p = Paginator(obj_list,page_count)
        instance = p.page(page)
        total_page = p.num_pages
        total_documents = p.count

        for obj in instance:# 볼드 처리해야할 단어들이 내용에 몇번 포함되어 있는지 계산
            x = 0
            application_number = str(obj.application_number)
            content = obj.content
            for n in bold_nouns:
                if n in application_number:
                    x += 1
                if content:
                    x += obj.content.count(n)
            if x != 0:
                direct_query.append([obj, x])
        direct_query = sorted(direct_query, key=lambda item: item[1], reverse=True)  # 직접 검색 결과(정렬된 것)
        data = SearchResultSerializer([item[0] for item in direct_query], many=True).data

        for idx, value in enumerate(data):
            value['key'] = idx

        result = {
            'total_number': len(direct_query),
            'bold': bold_nouns,
            'data': data,
            'page': page,
            'page_count': page_count,
            'total_page': total_page,
            'total_documents': total_documents
        }
        return Response(
            status=status.HTTP_200_OK,
            data=GeneralResponse(
                200,
                'search, OnlySearchDoc, get',
                result
            ).response
        )

class OnlySearchDocFixView(APIView):
    permission_classes = (AllowAny,)

    def get(self, request):
        start_time = datetime.now()
        y = time.time()
        nouns = request.GET.get('nouns')  # 검색어 받기
        start_date = request.GET.get('start_date')  # 날짜 시작 받기
        end_date = request.GET.get('end_date')  # 날짜 끝 받기
        category = request.GET.get('category')  # 카테고리가 있으면 받기
        page = request.GET.get('page')  # 몇 페이지에 해당하는지 받기
        page_count = request.GET.get('page_count')  # 한 페이지에 얼마나 보여줄지 받기
        session = request.GET.get('session_id')  # 검색하는 사람의 세션 받기
        interaction = 1 if any('&' in s for s in nouns) else 0
        nouns_list = nouns.split('&')  # 검색어가 &를 포함하고 있으면 분리
        nn = [x.strip() for x in nouns_list]  # blank(띄워쓰기) 삭제
        search_word = [lib_konlpy.ext_nouns(n) for n in nn]  # 명사만 추출하기 ex) 제조 금속 & 나노 전자 & 유전자 -> [['제조','금속'],['나노','전자'],[유전자]]
        list_nouns = lib_konlpy.ext_nouns(nouns)  # &에 관계없이 명사추출

        mecab = Mecab()
        if '&' in nouns:
            split_nouns = nouns.split('&')  # 검색어에 & 있으면 분리
        else:
            split_nouns = [nouns]  # 검색어에 & 없으면 형태에 맞게 변형

        print(datetime.now() - start_time)

        nouns = []
        for i in split_nouns:  # 검색어를 &로 나눈 후 형태소 분석 하였을때, 고유명사, 명사, 숫자, 영어가 있는 확인 후 nouns에 저장
            nn = []
            x = mecab.pos(i)
            for a in x:
                if a[1] == 'NNG' or a[1] == 'NNP' or a[1] == 'SN' or a[1] == 'SL':
                    nn.append(a[0])
            nouns.append(nn)
            del x

        bold_nouns = [item for items in nouns for item in items]  # 볼드 처리해야할 단어들

        obj_list = []
        for i in nouns:  # nouns를 돌면서 고유명사, 명사, 숫자, 영어를 포함하는 object들을 들고옴
            for num, j in enumerate(i):
                try:
                    application_number = int(j)
                    # obj_content = Document.objects.filter(
                    #     Q(content__icontains=j) | Q(title_english__icontains=j) | Q(application_number__icontains=application_number))
                    obj_content = Document.objects.defer('claim', 'legal_status', 'judging_progress_status', 'quotation', 'cited', 'pdf_path',
                                                         'html_path', 'figure_path', 'image', 'pdf', 'html').filter(
                        Q(content__icontains=j) | Q(title_english__icontains=j) | Q(application_number__icontains=application_number))
                except:
                    # obj_content = Document.objects.filter(Q(content__icontains=j) | Q(title_english__icontains=j))
                    obj_content = Document.objects.defer('claim', 'legal_status', 'judging_progress_status', 'quotation', 'cited', 'pdf_path', 'html_path', 'figure_path', 'image', 'pdf', 'html').filter(Q(content__icontains=j) | Q(title_english__icontains=j))
                if num == 0:
                    obj = obj_content
                else:
                    obj = obj | obj_content  # 하나의 형태소에 의해 검색된 모든 쿼리를 or 연산 처리 후 obj에 저장
            if not obj_list:
                obj_list = obj  # 제조 1887 & 나노 차금강 & 유전자 -> [[제조에 의해 검색된 쿼리셋, 1887에 의해 검색된 쿼리셋],[나노에 의해 검색된 쿼리셋,차금강에 의해 검색된 쿼리셋],[유전자에 의해 검색된 쿼리셋]]
            else:
                obj_list &= obj

        direct_query = []
        print(datetime.now() - start_time)
        for obj in obj_list.iterator():  # 볼드 처리해야할 단어들이 내용에 몇번 포함되어 있는지 계산
            x = 0
            application_number = str(obj.application_number)
            content = obj.content
            for n in bold_nouns:
                if n in application_number:
                    x += 1
                x += content.count(n)
            if x != 0:
                direct_query.append([obj, x])
        print(datetime.now() - start_time)
        direct_query = sorted(direct_query, key=lambda item: item[1], reverse=True)  # 직접 검색 결과(정렬된 것)

        p = Paginator(direct_query, page_count)
        instance = p.page(page)
        total_page = p.num_pages
        total_documents = p.count

        data = SearchResultSerializer([item[0] for item in instance], many=True).data

        for idx, value in enumerate(data):
            value['key'] = idx

        result = {
            'total_number': len(direct_query),
            'bold': bold_nouns,
            'data': data,
            'page': page,
            'page_count': page_count,
            'total_page': total_page,
            'total_documents': total_documents
        }

        print(datetime.now() - start_time)
        return Response(
            status=status.HTTP_200_OK,
            data=GeneralResponse(
                200,
                'search, OnlySearchDoc, get',
                result
            ).response
        )


class SearchRecommendDoc(APIView):  # 검색하였을 경우 추천 및 직접 검색에 해당하는 모든 컨텐츠를 보여주는 것
    permission_classes = (AllowAny,)

    def get(self, request):
        start_time = datetime.now()

        nouns = request.GET.get('nouns')  # 검색어 받기
        start_date = request.GET.get('start_date')  # 날짜 시작 받기
        end_date = request.GET.get('end_date')  # 날짜 끝 받기
        category = request.GET.get('category')  # 카테고리가 있으면 받기
        page = request.GET.get('page')  # 몇 페이지에 해당하는지 받기
        page_count = request.GET.get('page_count')  # 한 페이지에 얼마나 보여줄지 받기
        session = request.GET.get('session_id')  # 검색하는 사람의 세션 받기
        interaction = 1 if any('&' in s for s in nouns) else 0
        nouns_list = nouns.split('&')  # 검색어가 &를 포함하고 있으면 분리
        nn = [x.strip() for x in nouns_list]  # blank(띄워쓰기) 삭제
        search_word = [lib_konlpy.ext_nouns(n) for n in nn]  # 명사만 추출하기 ex) 제조 금속 & 나노 전자 & 유전자 -> [['제조','금속'],['나노','전자'],[유전자]]
        list_nouns = lib_konlpy.ext_nouns(nouns)  # &에 관계없이 명사추출

        if not page:
            page = 1
        if not page_count:
            page_count = 10

        mecab = Mecab()
        if '&' in nouns:
            split_nouns = nouns.split('&')  # 검색어에 & 있으면 분리
        else:
            split_nouns = [nouns]  # 검색어에 & 없으면 형태에 맞게 변형

        nouns = []
        for i in split_nouns:  # 검색어를 &로 나눈 후 형태소 분석 하였을때, 고유명사, 명사, 숫자, 영어가 있는 확인 후 nouns에 저장
            nn = []
            x = mecab.pos(i)
            for a in x:
                if a[1] == 'NNG' or a[1] == 'NNP' or a[1] == 'SN' or a[1] == 'SL':
                    nn.append(a[0])
            nouns.append(nn)
            del x

        bold_nouns = []  # 볼드 처리해야할 단어들
        for item in nouns:
            bold_nouns.extend(item)

        session_obj = Userid.objects.get(session=session)  # 사용자의 세션에 해당하는 object들고옴
        recommand_query = []
        try:
            last_nouns_obj = Trajectory.objects.filter(session=session_obj).last()  # 가장 최근에 경로를 가져옴(Trajectory)
            if last_nouns_obj:  # 만약 마지막 경로가 있다면
                last_nouns = last_nouns_obj.destination  # destination을 가져옴
                obj = Trajectory.objects.filter(departure=last_nouns)  # destination에 해당하는 trajectory를 모두 가져옴
                destination_number = obj.count()  # 해당하는 object들의 개수를 구함
                destination_nouns = [x.destination for x in obj]  # destination만 가져와서 list에 할당
                destination_dict = dict(Counter(destination_nouns))  # dict로 변환
                destination_dict_list = [[key, float(destination_dict[key] / destination_number)] for key in
                                         destination_dict.keys()]  # destination의 경로 확률을 구한 후 [[destination명사, 확률],[]...]로 저장

                for i in destination_dict_list:  # destination_list를 돌면서 [document.id, 경로확률 * 선호도]로 저장
                    info_list = Info_Document.objects.filter(document__nouns=i[0])
                    recommand_query = [(info.id, i[1] * info.score) for info in info_list.iterator()]  # recommand_query는 경로에 의한 추천 결과
        except:
            print('error')

        print(datetime.now() - start_time)

        model = Doc2Vec.load('model_doc.doc2vec')
        vector = model.infer_vector(nouns_list)  # 검색어에 포함되는 명사들의 벡터화(이하 검색어벡터)
        objs = docu2vec.objects.all().select_related('document')  # docu2vec(각 문서들의 벡터(이하 문서벡터))이 저장된 Object 들고옴

        similar_list = [[knn.cos_sim(vector, json.loads(ob.vector)), ob.document.application_number] for ob in
                        objs.iterator()]  # 검색어벡터와 문서벡터의 cos_sim을 구하고 application_number를 저장
        most_relative_list = sorted(similar_list, key=lambda x: x[0])  # cos_sim이 높은 순서대로 정렬
        most_relative_list = [x[1] for x in most_relative_list][:20]  # 제일 높은 20개만 가져온 후 application_number 저장
        most_relative_list = Document.objects.filter(application_number__in=most_relative_list)  # application_number순서대로 가져옴
        most_relative_list = [[i.id, np.clip(np.random.normal(0.001, 0.003, 1)[0], a_min=0.0, a_max=1.0)] for i in
                              most_relative_list]  # 검색어벡터에 의한 연관성 추천은 실제로 의미 있는것이 아니므로 적당한 난수를 만들어 점수로 할당                               # 단어 연관성에 의한 추천 리스트

        for relative in most_relative_list:
            recommand_query.append(relative)  # 경로 + 검색어연관성에 의한 추천을 모두 합한 리스트

        most_relative_id = [x[0] for x in recommand_query]  # 검색결과 및 추천결과를 most_relative_id에 할당
        score_ = [x[1] for x in recommand_query]  # most_relative_id의 순서에 맞게 점수를 score_에 할당
        result_ = most_relative_id  # result_에 다시 most_relative_id를 할당(헷갈리는것 방지)

        recommand_query.sort(key=itemgetter(1))
        recommand_query.reverse()  # direct 리스트를 점수로 정렬

        result_id = [x[0] for x in recommand_query]  # 컨텐츠만 다시 result_id에 저장
        res = []
        for re in result_id:
            if not re in res:
                res.append(re)  # 중복되지 않도록 result_id를 res에 다시 저장

        print(datetime.now() - start_time)

        instance = Document.objects.filter(id__in=[id for id, score in recommand_query])

        # p = Paginator(instance, page_count)
        # instance = p.page(page)
        # total_page = p.num_pages
        # total_documents = p.count

        instance = sorted(instance, key=lambda x: res.index(x.id))
        data = []  # 이후 아래는 필요한 정보를 dict에 담아서 list에 붙여 넣기 한후 response에 넘겨줌
        reco_num = 0
        for z, i in enumerate(instance):
            d = {}
            index = []
            summary = i.summary
            for no in list_nouns:
                n = []
                if no in summary:
                    a = summary.find(no)
                    n.append(a)
                    while summary[a + 1:].find(no) != -1:
                        a = summary[a + 1:].find(no) + a + 1
                        n.append(a)
                index.append(n)

            first_index = []
            for q in index:
                if len(q) != 0:
                    first_index.append(q[0])

            for p in range(len(first_index)):
                if len(first_index) == 1:
                    if first_index[0] < 90:
                        d['summary'] = summary[:180] + '...'
                    elif first_index[0] + 90 >= len(summary):
                        d['summary'] = '...' + summary[-180:]
                    else:
                        d['summary'] = '...' + summary[first_index[0] - 90:first_index[0] + 90] + '...'

                elif len(first_index) > 1:
                    qp = [first_index[0]]
                    for m in range(len(first_index)):
                        for n in range(len(first_index)):
                            if n > m:
                                if abs(first_index[m] - first_index[n]) < 90:
                                    qp.append(first_index[m])
                                else:
                                    qp.append(first_index[n])

            if len(first_index) > 1:
                qp = list(set(qp))
                sentence_size = int(180 / len(qp))
                qpp = []
                for x in range(len(qp)):
                    if first_index[x] < int(sentence_size / 2):
                        xxx = 1
                        qpp.append(summary[:sentence_size] + '...')
                    elif first_index[x] + int(sentence_size / 2) >= len(summary):
                        xxx = 2
                        qpp.append(summary[-sentence_size:])
                    else:
                        xxx = 3
                        qpp.append(
                            '...' + summary[first_index[0] - int(sentence_size / 2):first_index[0] + int(sentence_size / 2)] + '...')

                a = ''
                for pp in qpp:
                    a += pp
                d['summary'] = a
                del a, pp, qpp

            if i.id in result_:
                d['aivory'] = True
                d['score'] = score_[result_.index(i.id)]
                reco_num += 1
            else:
                print('not aivory')
                d['aivory'] = False
                d['score'] = 0

            d['key'] = z
            d['title_korean'] = i.title_korean
            d['title_english'] = i.title_english
            d['applicant'] = i.applicant.replace('|', ',')[:-1]
            d['application_number'] = i.application_number
            d['application_date'] = i.application_date
            d['registration_number'] = i.registration_number
            d['registration_date'] = i.registration_date

            if i.new_search_group == None:
                d['new_search_group'] = ''
                d['manager'] = []

            else:
                group_list = json.loads(i.new_search_group)
                xxs = SearchGroup.objects.filter(id__in=group_list)
                manager_id = []
                for xx in xxs:
                    a = json.loads(xx.manager)
                    manager_id_list = Manager.objects.filter(id__in=a)
                    mana = [x.id for x in manager_id_list]
                    manager_id.append(mana)
                group = [x.name for x in xxs]
                iix = ''
                for ix in group:
                    iix += ix + ','
                d['new_search_group'] = iix[:-1]

                manager_id = sum(manager_id, [])
                asdf = []
                for iff in manager_id:
                    if not iff in asdf:
                        asdf.append(iff)
                manager_id = asdf
                empty = []
                for manager in manager_id:
                    mana = Manager.objects.get(id=manager)
                    ddd = {'name': mana.name, 'phone': mana.phone, 'mail': mana.mail}
                    empty.append(ddd)
                d['manager'] = empty

            d['transfer_mail'] = i.tech_transfer_mail  #
            d['transfer_phone'] = i.tech_transfer_phone  #
            d['transfer_name'] = i.tech_transfer_name  #
            d['search_group'] = i.search_group
            data.append(d)
            del d
        print(datetime.now() - start_time)

        result = {
            'data': data,
            'total_number': len(data),
            'bold': bold_nouns,
            # 'page': page,
            # 'page_count': page_count,
            # 'total_page': total_page,
            # 'total_documents': total_documents
        }
        del data
        del model
        del mecab
        print('end_time: ', datetime.now() - start_time)
        return Response(
            status=status.HTTP_200_OK,
            data=GeneralResponse(
                200,
                'search_doc, SearchDoc, get',
                result
            ).response
        )


class SearchDoc(APIView):  # 검색하였을 경우 추천 및 직접 검색에 해당하는 모든 컨텐츠를 보여주는 것
    permission_classes = (AllowAny,)

    def get(self, request):
        start_time = datetime.now()

        nouns = request.GET.get('nouns')  # 검색어 받기
        start_date = request.GET.get('start_date')  # 날짜 시작 받기
        end_date = request.GET.get('end_date')  # 날짜 끝 받기
        category = request.GET.get('category')  # 카테고리가 있으면 받기
        page_number = int(request.GET.get('page_number'))  # 몇 페이지에 해당하는지 받기
        contain_number = int(request.GET.get('contain_number'))  # 한 페이지에 얼마나 보여줄지 받기
        session = request.GET.get('session_id')  # 검색하는 사람의 세션 받기
        interaction = 1 if any('&' in s for s in nouns) else 0
        nouns_list = nouns.split('&')  # 검색어가 &를 포함하고 있으면 분리
        nn = [x.strip() for x in nouns_list]  # blank(띄워쓰기) 삭제
        search_word = [lib_konlpy.ext_nouns(n) for n in nn]  # 명사만 추출하기 ex) 제조 금속 & 나노 전자 & 유전자 -> [['제조','금속'],['나노','전자'],[유전자]]
        list_nouns = lib_konlpy.ext_nouns(nouns)  # &에 관계없이 명사추출

        mecab = Mecab()
        if '&' in nouns:
            split_nouns = nouns.split('&')  # 검색어에 & 있으면 분리
        else:
            split_nouns = [nouns]  # 검색어에 & 없으면 형태에 맞게 변형

        nouns = []
        for i in split_nouns:  # 검색어를 &로 나눈 후 형태소 분석 하였을때, 고유명사, 명사, 숫자, 영어가 있는 확인 후 nouns에 저장
            nn = []
            x = mecab.pos(i)
            for a in x:
                if a[1] == 'NNG' or a[1] == 'NNP' or a[1] == 'SN' or a[1] == 'SL':
                    nn.append(a[0])
            nouns.append(nn)
            del x
        obj_list = []
        for i in nouns:  # nouns를 돌면서 고유명사, 명사, 숫자, 영어를 포함하는 object들을 들고옴
            for num, j in enumerate(i):
                if num == 0:
                    obj_content = Document.objects.filter(content__icontains=j)
                    obj_english = Document.objects.filter(title_english__icontains=j)
                    try:
                        obj_application = Document.objects.filter(application_number__icontains=int(j))
                    except:
                        obj_application = Document.objects.filter()
                    obj = obj_content | obj_english | obj_application
                else:
                    obj_content = obj | Document.objects.filter(content__icontains=j)
                    obj_english = Document.objects.filter(title_english__icontains=j)
                    try:
                        obj_application = Document.objects.filter(application_number__icontains=int(j))
                    except:
                        obj_application = Document.objects.filter()
                    obj = obj | obj_content | obj_english | obj_application  # 하나의 형태소에 의해 검색된 모든 쿼리를 or 연산 처리 후 obj에 저장
            obj_list.append(
                obj)  # 제조 1887 & 나노 차금강 & 유전자 -> [[제조에 의해 검색된 쿼리셋, 1887에 의해 검색된 쿼리셋],[나노에 의해 검색된 쿼리셋,차금강에 의해 검색된 쿼리셋],[유전자에 의해 검색된 쿼리셋]]

        search_query = Document.objects.all()
        for i in obj_list:
            if len(i) != 0:
                print(len(i))
                search_query &= i  # 리스트로 묶여 있는 obj_list를 for 돌면서 쿼리 형태로 바꿈

        bold_nouns = []  # 볼드 처리해야할 단어들
        for item in nouns:
            bold_nouns.extend(item)

        direct_query = []
        total_count = 0
        for obj in search_query:  # 볼드 처리해야할 단어들이 내용에 몇번 포함되어 있는지 계산
            x = 0
            for n in bold_nouns:
                if n in str(obj.application_number):
                    total_count += 1
                    x += 1
                total_count += obj.content.count(n)
                x += obj.content.count(n)
            if x != 0:
                direct_query.append([obj.id, x])
        direct_query.sort(key=itemgetter(1))
        direct_query.reverse()  # 직접 검색 결과(정렬된 것)
        if len(direct_query) == 0:
            return Response(
                status=status.HTTP_200_OK,
                data=GeneralResponse(
                    200,
                    'no result, SearchDoc, get',
                    {}
                ).response
            )

        session_obj = Userid.objects.get(session=session)  # 사용자의 세션에 해당하는 object들고옴
        recommand_query = []
        try:
            last_nouns_obj = Trajectory.objects.filter(session=session_obj).last()  # 가장 최근에 경로를 가져옴(Trajectory)
            if last_nouns_obj:  # 만약 마지막 경로가 있다면
                last_nouns = last_nouns_obj.destination  # destination을 가져옴
                obj = Trajectory.objects.filter(departure=last_nouns)  # destination에 해당하는 trajectory를 모두 가져옴
                destination_number = obj.count()  # 해당하는 object들의 개수를 구함
                destination_nouns = [x.destination for x in obj]  # destination만 가져와서 list에 할당
                destination_dict = dict(Counter(destination_nouns))  # dict로 변환
                destination_dict_list = [[key, float(destination_dict[key] / destination_number)] for key in
                                         destination_dict.keys()]  # destination의 경로 확률을 구한 후 [[destination명사, 확률],[]...]로 저장

                for i in destination_dict_list:  # destination_list를 돌면서 [document.id, 경로확률 * 선호도]로 저장
                    ob = Document.objects.filter(nouns=i[0])
                    for o in ob:
                        inf = Info_Document.objects.get(document=o)
                        recommand_query.append([inf.id, i[1] * inf.score])  # recommand_query는 경로에 의한 추천 결과
        except:
            print('error')

        model = Doc2Vec.load('model_doc.doc2vec')
        vector = model.infer_vector(nouns_list)  # 검색어에 포함되는 명사들의 벡터화(이하 검색어벡터)
        objs = docu2vec.objects.all()  # docu2vec(각 문서들의 벡터(이하 문서벡터))이 저장된 Object 들고옴

        similar_list = [[knn.cos_sim(vector, json.loads(ob.vector)), ob.document.application_number] for ob in
                        objs]  # 검색어벡터와 문서벡터의 cos_sim을 구하고 application_number를 저장
        most_relative_list = sorted(similar_list, key=lambda x: x[0])  # cos_sim이 높은 순서대로 정렬
        most_relative_list = [x[1] for x in most_relative_list][:20]  # 제일 높은 20개만 가져온 후 application_number 저장
        most_relative_list = Document.objects.filter(application_number__in=most_relative_list)  # application_number순서대로 가져옴
        most_relative_list = [[i.id, np.clip(np.random.normal(0.001, 0.003, 1)[0], a_min=0.0, a_max=1.0)] for i in
                              most_relative_list]  # 검색어벡터에 의한 연관성 추천은 실제로 의미 있는것이 아니므로 적당한 난수를 만들어 점수로 할당                               # 단어 연관성에 의한 추천 리스트

        for relative in most_relative_list:
            recommand_query.append(relative)  # 경로 + 검색어연관성에 의한 추천을 모두 합한 리스트

        direct = []
        for i in direct_query:  # direct라는 리스트에 직접검색결과와 추천결과를 모두 하나의 list에 넣음 [[컨텐츠, 점수],[]...]
            direct.append([i[0], float(i[1] / total_count)])
        for rec in recommand_query:
            direct.append(rec)

        most_relative_id = [x[0] for x in most_relative_list]  # 검색결과 및 추천결과를 most_relative_id에 할당
        score_ = [x[1] for x in most_relative_list]  # most_relative_id의 순서에 맞게 점수를 score_에 할당
        result_ = most_relative_id  # result_에 다시 most_relative_id를 할당(헷갈리는것 방지)

        direct.sort(key=itemgetter(1))
        direct.reverse()  # direct 리스트를 점수로 정렬

        result_id = [x[0] for x in recommand_query]  # 컨텐츠만 다시 result_id에 저장
        res = []
        for re in result_id:
            if not re in res:
                res.append(re)  # 중복되지 않도록 result_id를 res에 다시 저장

        result_object = list(Info_Document.objects.filter(id__in=res))
        result_object.sort(key=lambda t: res.index(t.pk))
        result_objs = result_object  # result_objs에 모든 결과를 점수 순서대로 쿼리셋 가져옴

        data = []  # 이후 아래는 필요한 정보를 dict에 담아서 list에 붙여 넣기 한후 response에 넘겨줌
        reco_num = 0
        for z, i in enumerate(result_objs):
            d = {}
            index = []
            for no in list_nouns:
                n = []
                if no in i.document.summary:
                    a = i.document.summary.find(no)
                    n.append(a)
                    while i.document.summary[a + 1:].find(no) != -1:
                        a = i.document.summary[a + 1:].find(no) + a + 1
                        n.append(a)
                index.append(n)

            first_index = []
            for q in index:
                if len(q) != 0:
                    first_index.append(q[0])

            for p in range(len(first_index)):
                if len(first_index) == 1:
                    if first_index[0] < 90:
                        d['summary'] = i.document.summary[:180] + '...'
                    elif first_index[0] + 90 >= len(i.document.summary):
                        d['summary'] = '...' + i.document.summary[-180:]
                    else:
                        d['summary'] = '...' + i.document.summary[first_index[0] - 90:first_index[0] + 90] + '...'

                elif len(first_index) > 1:
                    qp = [first_index[0]]
                    for m in range(len(first_index)):
                        for n in range(len(first_index)):
                            if n > m:
                                if abs(first_index[m] - first_index[n]) < 90:
                                    qp.append(first_index[m])
                                else:
                                    qp.append(first_index[n])

            if len(first_index) > 1:
                qp = list(set(qp))
                sentence_size = int(180 / len(qp))
                qpp = []
                for x in range(len(qp)):
                    if first_index[x] < int(sentence_size / 2):
                        xxx = 1
                        qpp.append(i.document.summary[:sentence_size] + '...')
                    elif first_index[x] + int(sentence_size / 2) >= len(i.document.summary):
                        xxx = 2
                        qpp.append(i.document.summary[-sentence_size:])
                    else:
                        xxx = 3
                        qpp.append(
                            '...' + i.document.summary[first_index[0] - int(sentence_size / 2):first_index[0] + int(sentence_size / 2)] + '...')

                a = ''
                for pp in qpp:
                    a += pp
                d['summary'] = a
                del a, pp, qpp

            if i.document.id in result_:
                d['aivory'] = True
                d['score'] = score_[result_.index(i.document.id)]
                reco_num += 1
            else:
                d['aivory'] = False
                d['score'] = 0
            d['key'] = z
            d['title_korean'] = i.document.title_korean
            d['title_english'] = i.document.title_english
            d['applicant'] = i.document.applicant.replace('|', ',')[:-1]
            d['application_number'] = i.document.application_number
            d['application_date'] = i.document.application_date
            d['registration_number'] = i.document.registration_number
            d['registration_date'] = i.document.registration_date

            if i.document.new_search_group == None:
                d['new_search_group'] = ''
                d['manager'] = []

            else:
                group_list = json.loads(i.document.new_search_group)
                xxs = SearchGroup.objects.filter(id__in=group_list)
                manager_id = []
                for xx in xxs:
                    a = json.loads(xx.manager)
                    manager_id_list = Manager.objects.filter(id__in=a)
                    mana = [x.id for x in manager_id_list]
                    manager_id.append(mana)
                group = [x.name for x in xxs]
                iix = ''
                for ix in group:
                    iix += ix + ','
                d['new_search_group'] = iix[:-1]

                manager_id = sum(manager_id, [])
                asdf = []
                for iff in manager_id:
                    if not iff in asdf:
                        asdf.append(iff)
                manager_id = asdf
                empty = []
                for manager in manager_id:
                    mana = Manager.objects.get(id=manager)
                    ddd = {'name': mana.name, 'phone': mana.phone, 'mail': mana.mail}
                    empty.append(ddd)
                d['manager'] = empty

            d['transfer_mail'] = i.document.tech_transfer_mail  #
            d['transfer_phone'] = i.document.tech_transfer_phone  #
            d['transfer_name'] = i.document.tech_transfer_name  #
            d['search_group'] = i.document.search_group
            data.append(d)
            del d
        result = {}
        result['data'] = data
        result['total_number'] = len(result_objs)
        result['recom_number'] = reco_num
        result['bold'] = bold_nouns
        del data
        del model
        del mecab
        print('end_time: ', datetime.now() - start_time)
        return Response(
            status=status.HTTP_200_OK,
            data=GeneralResponse(
                200,
                'search_doc, SearchDoc, get',
                result
            ).response
        )


class SaveSearchWord(APIView):  # DB에 검색어가 언제, 누구에게서 검색되었는지 저장
    permission_classes = (AllowAny,)

    def post(self, request):
        try:
            session = request.data.get('session_id')
            word = request.data['info']
            now_time = timezone.now()
            session_word = Search_Word_Path(session=session, search_date=now_time, search_word=word)

            session_word.save()
            return Response(
                status=status.HTTP_200_OK,
                data=GeneralResponse(
                    200,
                    'search word is saved well, SaveSearchWord, post',
                    {}
                ).response
            )

        except Exception as e:
            log.debug(e)
            return Response(
                status=status.HTTP_400_BAD_REQUEST,
                data=GeneralResponse(
                    400,
                    'search word is not saved well, SaveSearchWord, post',
                    {}
                ).response
            )


class CreateDateInformation(APIView):  # 어떤 컨텐츠가 언제 읽혔는지 DB(Consumed_contents에 저장)
    permission_classes = (AllowAny,)

    def post(self, request, pk):
        docu = Document.objects.get(application_number=pk)
        now_time = datetime.now()
        date_info = Consumed_contents(document=docu, consumed_time=now_time)
        date_info.save()
        return Response(
            status=status.HTTP_200_OK,
            data=GeneralResponse(
                200,
                'createdateinformation, CreateDateInformation, post',
                {}
            ).response
        )


class Clicked(APIView):  # 컨텐츠가 클릭되었을 때 Info_Document 갱신
    permission_classes = (AllowAny,)

    def put(self, request, pk, format=None):
        try:
            obj = Document.objects.get(application_number=pk)
            info = Info_Document.objects.get(document=obj)
            info.clicked += 1
            b = beta.rvs(info.clicked, info.recommended)
            info.score = b
            info.save()
            return Response(
                status=status.HTTP_200_OK,
                data=GeneralResponse(
                    200,
                    'Click complete',
                    {}
                ).response
            )
        except Exception as e:
            log.debug(e)
            return Response(
                status=status.HTTP_400_BAD_REQUEST,
                data=GeneralResponse(
                    400,
                    'Click False',
                    {}
                ).response
            )


class Recommended(APIView):  # 추천할 때 Info_Document 갱신
    permission_classes = (AllowAny,)

    def put(self, request, format=None):
        try:
            patent_number_list = request.GET.get('list')
            patent_number_list = json.loads(patent_number_list)
            for patent_number in patent_number_list:
                obj = Document.objects.get(application_number=patent_number)
                info = Info_Document.objects.get(document=obj)
                info.recommended += 1
                b = beta.rvs(info.clicked, info.recommended)
                info.score = b
                info.save()
            return Response(
                status=status.HTTP_200_OK,
                data=GeneralResponse(
                    200,
                    'Recommend done',
                    {}
                ).response
            )
        except Exception as e:
            log.debug(e)
            return Response(
                status=status.HTTP_400_BAD_REQUEST,
                data=GeneralResponse(
                    400,
                    'Recommend False',
                    {}
                ).response
            )

#
# class TestAPI(APIView):
#     permission_classes = (AllowAny, )
#
#     def get(self, request):
#         with connection.cursor() as cursor:
#             cursor.execute('SELECT * FROM ipbiz_ibs.core_manager')
#             row = cursor.fetchall()
#             print(row)
#         return Response(row)
