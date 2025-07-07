'use client';

import React, { useEffect } from 'react';
import { Button, Typography, Row, Col, Card, Space, Statistic } from 'antd';
import { 
  UserOutlined, 
  TeamOutlined, 
  RocketOutlined, 
  ClockCircleOutlined, 
  ExperimentOutlined, 
  CheckCircleOutlined 
} from '@ant-design/icons';
import Image from 'next/image';
import { useRouter } from 'next/navigation';
import * as echarts from 'echarts';
import MainLayout from '@/components/layout/MainLayout';
import styles from './page.module.scss';

const { Title, Paragraph, Text } = Typography;

export default function Home() {
  const router = useRouter();

  useEffect(() => {
    // Initialize the chart
    const chartElement = document.getElementById('visualization-chart');
    if (chartElement) {
      const chart = echarts.init(chartElement);
      
      const option = {
        tooltip: {
          trigger: 'item',
          formatter: '{a} <br/>{b}: {c} ({d}%)'
        },
        legend: {
          data: ['求职者', '雇主', '专家', '完成匹配', '正在进行', '准备中']
        },
        series: [
          {
            name: '平台用户',
            type: 'pie',
            radius: ['50%', '70%'],
            avoidLabelOverlap: false,
            itemStyle: {
              borderRadius: 10,
              borderColor: '#fff',
              borderWidth: 2
            },
            label: {
              show: false,
              position: 'center'
            },
            emphasis: {
              label: {
                show: true,
                fontSize: '20',
                fontWeight: 'bold'
              }
            },
            labelLine: {
              show: false
            },
            data: [
              { value: 1048, name: '求职者' },
              { value: 735, name: '雇主' },
              { value: 580, name: '专家' }
            ]
          },
          {
            name: '合作状态',
            type: 'pie',
            radius: ['20%', '40%'],
            avoidLabelOverlap: false,
            itemStyle: {
              borderRadius: 10,
              borderColor: '#fff',
              borderWidth: 2
            },
            label: {
              show: false,
              position: 'center'
            },
            emphasis: {
              label: {
                show: true,
                fontSize: '20',
                fontWeight: 'bold'
              }
            },
            labelLine: {
              show: false
            },
            data: [
              { value: 1048, name: '完成匹配' },
              { value: 735, name: '正在进行' },
              { value: 580, name: '准备中' }
            ]
          }
        ]
      };

      chart.setOption(option);
      
      const resizeHandler = () => {
        chart.resize();
      };
      
      window.addEventListener('resize', resizeHandler);
      
      return () => {
        window.removeEventListener('resize', resizeHandler);
        chart.dispose();
      };
    }
  }, []);

  return (
    <MainLayout>
      <div className={styles.heroSection}>
        <div className={styles.heroContent}>
          <Title className={styles.heroTitle}>
            个人数字分身平台
          </Title>
          <Paragraph className={styles.heroSubtitle}>
            赋能个体，连接未来。打造您的智能数字分身，让知识和经验7x24小时创造价值。
          </Paragraph>
          <Space size="large">
            <Button 
              type="primary" 
              size="large" 
              onClick={() => router.push('/register')}
              className={styles.primaryButton}
            >
              立即注册
            </Button>
            <Button 
              size="large" 
              onClick={() => router.push('/features')}
              className={styles.secondaryButton}
            >
              了解更多
            </Button>
          </Space>
        </div>
        <div className={styles.heroImageContainer}>
          <div className={styles.heroImage}>
            {/* Replace with your own hero image */}
            <div className={styles.heroImagePlaceholder} />
          </div>
        </div>
      </div>

      <div className={styles.statsSection}>
        <Row gutter={[24, 24]}>
          <Col xs={12} sm={12} md={8} lg={6} xl={6}>
            <Card className={styles.statCard}>
              <Statistic 
                title="注册用户" 
                value={2363} 
                prefix={<UserOutlined />} 
                valueStyle={{ color: '#1677ff' }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={12} md={8} lg={6} xl={6}>
            <Card className={styles.statCard}>
              <Statistic 
                title="企业用户" 
                value={735} 
                prefix={<TeamOutlined />} 
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={12} md={8} lg={6} xl={6}>
            <Card className={styles.statCard}>
              <Statistic 
                title="专家用户" 
                value={580} 
                prefix={<ExperimentOutlined />} 
                valueStyle={{ color: '#722ed1' }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={12} md={8} lg={6} xl={6}>
            <Card className={styles.statCard}>
              <Statistic 
                title="成功匹配" 
                value={1248} 
                prefix={<CheckCircleOutlined />} 
                valueStyle={{ color: '#fa8c16' }}
              />
            </Card>
          </Col>
        </Row>
      </div>

      <div className={styles.featuresSection}>
        <Title level={2} className={styles.sectionTitle}>平台核心功能</Title>
        <Row gutter={[32, 32]}>
          <Col xs={24} sm={24} md={8} lg={8} xl={8}>
            <Card className={styles.featureCard}>
              <div className={styles.featureIcon}>
                <UserOutlined />
              </div>
              <Title level={4}>数字分身创建与养成</Title>
              <Paragraph>
                支持多种方式构建分身：简历上传、在线链接导入、手动补充。创建后立即进行对话测试，手动修正优化分身知识库。
              </Paragraph>
            </Card>
          </Col>
          <Col xs={24} sm={24} md={8} lg={8} xl={8}>
            <Card className={styles.featureCard}>
              <div className={styles.featureIcon}>
                <RocketOutlined />
              </div>
              <Title level={4}>智能求职与人才匹配</Title>
              <Paragraph>
                基于您的数字分身能力模型，智能推送最匹配的工作机会。雇主可先与候选人的数字分身对话，提高筛选效率。
              </Paragraph>
            </Card>
          </Col>
          <Col xs={24} sm={24} md={8} lg={8} xl={8}>
            <Card className={styles.featureCard}>
              <div className={styles.featureIcon}>
                <ClockCircleOutlined />
              </div>
              <Title level={4}>专家知识变现</Title>
              <Paragraph>
                专家可上传专业知识内容，通过数字分身提供7x24小时在线咨询，实现知识高效变现，同时增强个人品牌影响力。
              </Paragraph>
            </Card>
          </Col>
        </Row>
      </div>

      <div className={styles.visualizationSection}>
        <Title level={2} className={styles.sectionTitle}>平台数据可视化</Title>
        <div className={styles.visualizationContainer}>
          <div id="visualization-chart" className={styles.chart}></div>
        </div>
      </div>

      <div className={styles.ctaSection}>
        <Title level={2} className={styles.ctaTitle}>打造您的数字分身，开启智能职业生涯</Title>
        <Paragraph className={styles.ctaText}>
          无论您是求职者、雇主还是专家，我们的平台都能为您提供量身定制的解决方案。
        </Paragraph>
        <Button 
          type="primary" 
          size="large" 
          onClick={() => router.push('/register')}
          className={styles.ctaButton}
        >
          立即加入
        </Button>
      </div>
    </MainLayout>
  );
} 