import React, { useState, useEffect } from 'react';
import { Form, Input, Button, message, Row, Col, Card, Typography, Progress } from 'antd';
import { UserOutlined, LockOutlined, SafetyOutlined, ReloadOutlined } from '@ant-design/icons';
import { useNavigate, Link } from 'react-router-dom';
import { useLocalStorage } from '../../LocalStorageContext';

const { Title, Text } = Typography;

const Login: React.FC = () => {
  const [form] = Form.useForm();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [captchaId, setCaptchaId] = useState('');
  const [captchaUrl, setCaptchaUrl] = useState('');
  const [passwordStrength, setPasswordStrength] = useState(0);
  const [isLoggedIn, setIsLoggedIn] = useLocalStorage<boolean>('isLoggedIn', false);
  const [currentUser, setCurrentUser] = useLocalStorage<any>('currentUser', null);

  // 检查会话状态
  useEffect(() => {
    // 如果已经登录，直接跳转到仪表盘
    if (isLoggedIn && currentUser) {
      navigate('/dashboard');
      return;
    }

    const checkSession = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/session', {
          method: 'GET',
          credentials: 'include',
        });

        if (response.ok) {
          const data = await response.json();
          setIsLoggedIn(true);
          setCurrentUser(data.user);
          // 使用replace替代push，避免浏览器历史堆栈问题
          navigate('/dashboard', { replace: true });
        }
      } catch (error) {
        console.error('会话检查失败:', error);
      }
    };

    checkSession();
  }, [isLoggedIn, currentUser, navigate]);

  // 获取验证码
  const refreshCaptcha = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/captcha', {
        method: 'GET',
        credentials: 'include', // 包含cookies
      });
      
      if (!response.ok) {
        message.error(`获取验证码失败: ${response.status}`);
        return;
      }
      
      // 检查并获取验证码ID
      const captchaId = response.headers.get('captcha-id');
      
      if (captchaId) {
        setCaptchaId(captchaId);
        
        // 使用blob URL
        const blob = await response.blob();
        
        // 如果存在旧的URL，先释放它
        if (captchaUrl) {
          URL.revokeObjectURL(captchaUrl);
        }
        
        const url = URL.createObjectURL(blob);
        setCaptchaUrl(url);
      } else {
        message.error('未获取到验证码ID，请刷新页面重试');
      }
    } catch (error) {
      console.error('获取验证码失败:', error);
      message.error('获取验证码失败，请刷新页面重试');
    }
  };

  // 只在组件挂载时加载验证码
  useEffect(() => {
    // 如果未登录，才加载验证码
    if (!isLoggedIn) {
      refreshCaptcha();
    }
    
    // 组件卸载时清理blob URL
    return () => {
      if (captchaUrl) {
        URL.revokeObjectURL(captchaUrl);
      }
    };
  }, []);

  // 计算密码强度
  const calculatePasswordStrength = (password: string) => {
    if (!password) {
      setPasswordStrength(0);
      return;
    }

    let strength = 0;
    
    // 长度检查
    if (password.length >= 8) strength += 25;
    
    // 检查是否包含数字
    if (/\d/.test(password)) strength += 25;
    
    // 检查是否包含小写字母
    if (/[a-z]/.test(password)) strength += 25;
    
    // 检查是否包含大写字母或特殊字符
    if (/[A-Z]/.test(password) || /[^a-zA-Z0-9]/.test(password)) strength += 25;
    
    setPasswordStrength(strength);
  };

  // 处理表单提交
  const onFinish = async (values: any) => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: values.username,
          password: values.password,
          captcha: values.captcha,
          captcha_id: captchaId,
        }),
        credentials: 'include', // 重要：确保Cookie可以被设置
      });

      const data = await response.json();

      if (response.ok) {
        message.success('登录成功');
        setIsLoggedIn(true);
        setCurrentUser(data.user);
        // 使用replace模式，避免后退到登录页
        navigate('/dashboard', { replace: true });
      } else {
        message.error(data.detail || '登录失败');
        refreshCaptcha();
      }
    } catch (error) {
      console.error('登录请求失败:', error);
      message.error('登录请求失败，请稍后重试');
      refreshCaptcha();
    } finally {
      setLoading(false);
    }
  };

  // 密码强度指示器
  const renderPasswordStrength = () => {
    let color = '#ff4d4f';
    let text = '弱';
    
    if (passwordStrength >= 75) {
      color = '#52c41a';
      text = '强';
    } else if (passwordStrength >= 50) {
      color = '#1890ff';
      text = '中';
    } else if (passwordStrength >= 25) {
      color = '#faad14';
      text = '弱';
    }
    
    return (
      <div style={{ marginBottom: '20px' }}>
        <Progress percent={passwordStrength} showInfo={false} strokeColor={color} />
        <Text type="secondary">密码强度: {text}</Text>
      </div>
    );
  };

  // 如果已登录，直接返回null（不渲染登录表单）
  if (isLoggedIn && currentUser) {
    return null;
  }

  return (
    <Row justify="center" align="middle" style={{ minHeight: '100vh' }}>
      <Col xs={22} sm={16} md={12} lg={8}>
        <Card>
          <Title level={2} style={{ textAlign: 'center', marginBottom: '30px' }}>
            登录
          </Title>
          
          <Form
            form={form}
            name="login"
            onFinish={onFinish}
            autoComplete="off"
            layout="vertical"
          >
            <Form.Item
              name="username"
              rules={[{ required: true, message: '请输入用户名' }]}
            >
              <Input prefix={<UserOutlined />} placeholder="用户名" />
            </Form.Item>

            <Form.Item
              name="password"
              rules={[{ required: true, message: '请输入密码' }]}
            >
              <Input.Password 
                prefix={<LockOutlined />} 
                placeholder="密码"
                onChange={(e) => calculatePasswordStrength(e.target.value)}
              />
            </Form.Item>

            {renderPasswordStrength()}

            <Form.Item>
              <Row gutter={8}>
                <Col span={16}>
                  <Form.Item
                    name="captcha"
                    noStyle
                    rules={[{ required: true, message: '请输入验证码' }]}
                  >
                    <Input prefix={<SafetyOutlined />} placeholder="验证码" />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <div style={{ position: 'relative' }}>
                    {captchaUrl && (
                      <img
                        src={captchaUrl}
                        alt="验证码"
                        style={{ width: '100%', height: '32px', cursor: 'pointer' }}
                        onClick={refreshCaptcha}
                      />
                    )}
                    <Button
                      icon={<ReloadOutlined />}
                      size="small"
                      type="text"
                      onClick={refreshCaptcha}
                      style={{
                        position: 'absolute',
                        right: '0',
                        top: '0',
                        background: 'rgba(255, 255, 255, 0.7)',
                      }}
                    />
                  </div>
                </Col>
              </Row>
            </Form.Item>

            <Form.Item>
              <Button type="primary" htmlType="submit" loading={loading} block>
                登录
              </Button>
            </Form.Item>

            <Form.Item style={{ marginBottom: 0 }}>
              <Row justify="space-between">
                <Col>
                  <Link to="/forgot-password">忘记密码？</Link>
                </Col>
                <Col>
                  <Link to="/register">注册新账户</Link>
                </Col>
              </Row>
            </Form.Item>
          </Form>
        </Card>
      </Col>
    </Row>
  );
};

export default Login; 