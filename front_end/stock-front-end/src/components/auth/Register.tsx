import React, { useState, useEffect } from 'react';
import { Form, Input, Button, message, Row, Col, Card, Typography, Progress } from 'antd';
import { UserOutlined, LockOutlined, MailOutlined, SafetyOutlined, ReloadOutlined } from '@ant-design/icons';
import { useNavigate, Link } from 'react-router-dom';

const { Title, Text } = Typography;

const Register: React.FC = () => {
  const [form] = Form.useForm();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [captchaId, setCaptchaId] = useState('');
  const [captchaUrl, setCaptchaUrl] = useState('');
  const [passwordStrength, setPasswordStrength] = useState(0);

  // 获取验证码
  const refreshCaptcha = async () => {
    try {
      console.log('开始获取验证码');
      const response = await fetch('http://localhost:8000/api/captcha', {
        method: 'GET',
        credentials: 'include', // 包含cookies
      });
      
      console.log('验证码响应状态:', response.status);
      
      if (!response.ok) {
        console.error('验证码响应错误:', response.status);
        message.error(`获取验证码失败: ${response.status}`);
        return;
      }
      
      // 检查并获取验证码ID
      const captchaId = response.headers.get('captcha-id');
      console.log('获取到验证码ID:', captchaId);
      
      if (captchaId) {
        setCaptchaId(captchaId);
        
        // 使用blob URL
        const blob = await response.blob();
        
        // 如果存在旧的URL，先释放它
        if (captchaUrl) {
          URL.revokeObjectURL(captchaUrl);
        }
        
        const url = URL.createObjectURL(blob);
        console.log('创建验证码图片URL');
        setCaptchaUrl(url);
      } else {
        console.error('未获取到验证码ID');
        message.error('未获取到验证码ID，请刷新页面重试');
      }
    } catch (error) {
      console.error('获取验证码失败:', error);
      message.error('获取验证码失败，请刷新页面重试');
    }
  };

  // 初始加载验证码
  useEffect(() => {
    refreshCaptcha();
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
    if (values.password !== values.confirm) {
      message.error('两次输入的密码不一致');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: values.username,
          password: values.password,
          email: values.email,
          captcha: values.captcha,
          captcha_id: captchaId,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        message.success('注册成功，请登录');
        navigate('/login');
      } else {
        message.error(data.detail || '注册失败');
        refreshCaptcha();
      }
    } catch (error) {
      console.error('注册请求失败:', error);
      message.error('注册请求失败，请稍后重试');
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

  return (
    <Row justify="center" align="middle" style={{ minHeight: '100vh' }}>
      <Col xs={22} sm={16} md={12} lg={8}>
        <Card>
          <Title level={2} style={{ textAlign: 'center', marginBottom: '30px' }}>
            注册账户
          </Title>
          
          <Form
            form={form}
            name="register"
            onFinish={onFinish}
            autoComplete="off"
            layout="vertical"
          >
            <Form.Item
              name="username"
              rules={[
                { required: true, message: '请输入用户名' },
                { min: 4, message: '用户名至少4个字符' },
              ]}
            >
              <Input prefix={<UserOutlined />} placeholder="用户名" />
            </Form.Item>

            <Form.Item
              name="email"
              rules={[
                { type: 'email', message: '请输入有效的邮箱地址' },
                { required: true, message: '请输入邮箱' },
              ]}
            >
              <Input prefix={<MailOutlined />} placeholder="邮箱" />
            </Form.Item>

            <Form.Item
              name="password"
              rules={[
                { required: true, message: '请输入密码' },
                { min: 6, message: '密码至少6个字符' },
              ]}
            >
              <Input.Password 
                prefix={<LockOutlined />} 
                placeholder="密码"
                onChange={(e) => calculatePasswordStrength(e.target.value)}
              />
            </Form.Item>

            {renderPasswordStrength()}

            <Form.Item
              name="confirm"
              rules={[
                { required: true, message: '请确认密码' },
                ({ getFieldValue }) => ({
                  validator(_, value) {
                    if (!value || getFieldValue('password') === value) {
                      return Promise.resolve();
                    }
                    return Promise.reject(new Error('两次输入的密码不一致'));
                  },
                }),
              ]}
            >
              <Input.Password prefix={<LockOutlined />} placeholder="确认密码" />
            </Form.Item>

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
                注册
              </Button>
            </Form.Item>

            <Form.Item style={{ marginBottom: 0, textAlign: 'center' }}>
              已有账户? <Link to="/login">立即登录</Link>
            </Form.Item>
          </Form>
        </Card>
      </Col>
    </Row>
  );
};

export default Register; 