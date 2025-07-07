import React from 'react';
import { Layout, Row, Col, Space, Typography, Divider } from 'antd';
import Link from 'next/link';
import {
  GithubOutlined,
  LinkedinOutlined,
  TwitterOutlined,
  MailOutlined,
} from '@ant-design/icons';
import styles from './Footer.module.scss';

const { Footer: AntFooter } = Layout;
const { Title, Text, Link: AntLink } = Typography;

const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  return (
    <AntFooter className={styles.footer}>
      <div className={styles.footerContent}>
        <Row gutter={[24, 24]}>
          <Col xs={24} sm={24} md={8} lg={8} xl={8}>
            <div className={styles.footerSection}>
              <Title level={4} className={styles.sectionTitle}>数字协同</Title>
              <Text className={styles.sectionText}>
                赋能个体，连接未来。我们致力于为每一位专业人士打造智能的"数字分身"，让知识和经验能够7x24小时不间断地创造价值。
              </Text>
            </div>
          </Col>
          <Col xs={24} sm={12} md={5} lg={5} xl={5}>
            <div className={styles.footerSection}>
              <Title level={4} className={styles.sectionTitle}>关于我们</Title>
              <ul className={styles.footerLinks}>
                <li><Link href="/about">公司简介</Link></li>
                <li><Link href="/team">团队成员</Link></li>
                <li><Link href="/careers">加入我们</Link></li>
                <li><Link href="/contact">联系方式</Link></li>
              </ul>
            </div>
          </Col>
          <Col xs={24} sm={12} md={5} lg={5} xl={5}>
            <div className={styles.footerSection}>
              <Title level={4} className={styles.sectionTitle}>产品服务</Title>
              <ul className={styles.footerLinks}>
                <li><Link href="/features">功能特点</Link></li>
                <li><Link href="/pricing">定价方案</Link></li>
                <li><Link href="/case-studies">案例研究</Link></li>
                <li><Link href="/faq">常见问题</Link></li>
              </ul>
            </div>
          </Col>
          <Col xs={24} sm={24} md={6} lg={6} xl={6}>
            <div className={styles.footerSection}>
              <Title level={4} className={styles.sectionTitle}>联系我们</Title>
              <Space direction="vertical" size="small">
                <Text className={styles.contactInfo}>
                  <MailOutlined className={styles.contactIcon} /> contact@digitaltwin.com
                </Text>
                <Space className={styles.socialLinks}>
                  <a href="https://github.com" target="_blank" rel="noopener noreferrer">
                    <GithubOutlined className={styles.socialIcon} />
                  </a>
                  <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer">
                    <LinkedinOutlined className={styles.socialIcon} />
                  </a>
                  <a href="https://twitter.com" target="_blank" rel="noopener noreferrer">
                    <TwitterOutlined className={styles.socialIcon} />
                  </a>
                </Space>
              </Space>
            </div>
          </Col>
        </Row>
        <Divider className={styles.divider} />
        <div className={styles.bottomFooter}>
          <Text className={styles.copyright}>
            &copy; {currentYear} 数字协同人网页. 保留所有权利.
          </Text>
          <Space size="middle" className={styles.legalLinks}>
            <Link href="/terms">服务条款</Link>
            <Link href="/privacy">隐私政策</Link>
          </Space>
        </div>
      </div>
    </AntFooter>
  );
};

export default Footer; 