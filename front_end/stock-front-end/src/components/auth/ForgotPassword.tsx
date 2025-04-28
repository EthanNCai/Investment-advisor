import React, { useState, useEffect } from 'react';
import { Form, Input, Button, message, Row, Col, Card, Typography } from 'antd';
import { UserOutlined, MailOutlined, SafetyOutlined, ReloadOutlined } from '@ant-design/icons';
import { useNavigate, Link } from 'react-router-dom';

const { Title, Paragraph } = Typography;

const ForgotPassword: React.FC = () => {
  const [form] = Form.useForm();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [captchaId, setCaptchaId] = useState('');
  const [captchaUrl, setCaptchaUrl] = useState('');

  // 获取验证码
  const refreshCaptcha = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/captcha');
      const captchaId = response.headers.get('captcha-id');
      if (captchaId) {
        setCaptchaId(captchaId);
        // 使用blob URL
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        setCaptchaUrl(url);
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

  // 处理表单提交
  const onFinish = async (values: any) => {
    setLoading(true);
    try {
      // 显示成功信息
      message.success('重置密码请求已发送，请查看邮箱');
      
      // 重定向到登录页面
      setTimeout(() => {
        navigate('/login');
      }, 2000);
    } catch (error) {
      console.error('请求失败:', error);
      message.error('请求失败，请稍后重试');
      refreshCaptcha();
    } finally {
      setLoading(false);
    }
  };

  return (
    <Row justify="center" align="middle" style={{ minHeight: '100vh' }}>
      <Col xs={22} sm={16} md={12} lg={8}>
        <Card>
          <Title level={2} style={{ textAlign: 'center', marginBottom: '20px' }}>
            重置密码
          </Title>
          
          <Paragraph style={{ textAlign: 'center', marginBottom: '30px' }}>
            请输入您的账号和注册邮箱，我们会向您发送重置密码的链接。
          </Paragraph>
          
          <Form
            form={form}
            name="forgotPassword"
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
              name="email"
              rules={[
                { type: 'email', message: '请输入有效的邮箱地址' },
                { required: true, message: '请输入邮箱' },
              ]}
            >
              <Input prefix={<MailOutlined />} placeholder="注册邮箱" />
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
                提交
              </Button>
            </Form.Item>

            <Form.Item style={{ marginBottom: 0, textAlign: 'center' }}>
              <Link to="/login">返回登录</Link>
            </Form.Item>
          </Form>
        </Card>
      </Col>
    </Row>
  );
};

export default ForgotPassword; 