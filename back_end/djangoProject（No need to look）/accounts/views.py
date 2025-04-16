from django.contrib.auth import authenticate
from django.http import JsonResponse
from django.shortcuts import render
from rest_framework.authtoken.admin import User
from rest_framework.decorators import api_view
from rest_framework.authtoken.models import Token


# Create your views here.


def user_login(request):
    # user = get_object_or_404(User, username=request.data.get('username'))
    # user = User.objects.create_user("cdp", None, "123")
    # user.save()
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        if username and password:
            _user = authenticate(username=username, password=password)
            if _user is not None:
                return JsonResponse({'success': 'suc'})
            # 新用户登录
            user = User.objects.create_user(username=username, password=password)
            if user:
                token = Token.objects.create(user=user)
                with open('token.txt', 'a') as file:
                    file.write('\n' + str({username: str(token)}))
                    file.flush()
                    print("登录成功，并且 token 已保存到 token.txt 文件中。")
                    file.close()
                return JsonResponse({'token': token.key})
            else:
                return JsonResponse({'error': 'username or password cannot be empty'})


    else:
        return render(request, 'login.html')
