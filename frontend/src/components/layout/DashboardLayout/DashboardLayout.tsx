import React, { useState, ReactNode } from 'react';
import { Layout, Menu, Button, Avatar, Typography } from 'antd';
import {
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  UserOutlined,
  DesktopOutlined,
  FileOutlined,
  TeamOutlined,
  BarsOutlined,
  SettingOutlined,
  BookOutlined,
  MessageOutlined,
  DashboardOutlined,
} from '@ant-design/icons';
import Link from 'next/link';
import { useRouter } from 'next/router';
import Header from '../Header';
import Footer from '../Footer';
import styles from './DashboardLayout.module.scss';

const { Content, Sider } = Layout;
const { Title, Text } = Typography;

interface DashboardLayoutProps {
  children: ReactNode;
}

const DashboardLayout: React.FC<DashboardLayoutProps> = ({ children }) => {
  const [collapsed, setCollapsed] = useState(false);
  const router = useRouter();

  const toggleCollapsed = () => {
    setCollapsed(!collapsed);
  };

  const getSelectedKey = () => {
    const path = router.pathname;
    if (path.includes('/dashboard/digital-twin')) return 'digital-twin';
    if (path.includes('/dashboard/jobs')) return 'jobs';
    if (path.includes('/dashboard/candidates')) return 'candidates';
    if (path.includes('/dashboard/knowledge-base')) return 'knowledge-base';
    if (path.includes('/dashboard/settings')) return 'settings';
    if (path.includes('/dashboard/messages')) return 'messages';
    return 'dashboard';
  };

  const menuItems = [
    {
      key: 'dashboard',
      icon: <DashboardOutlined />,
      label: <Link href="/dashboard">仪表盘</Link>,
    },
    {
      key: 'digital-twin',
      icon: <DesktopOutlined />,
      label: <Link href="/dashboard/digital-twin">数字分身</Link>,
    },
    {
      key: 'jobs',
      icon: <FileOutlined />,
      label: <Link href="/dashboard/jobs">职位</Link>,
    },
    {
      key: 'candidates',
      icon: <TeamOutlined />,
      label: <Link href="/dashboard/candidates">候选人</Link>,
    },
    {
      key: 'knowledge-base',
      icon: <BookOutlined />,
      label: <Link href="/dashboard/knowledge-base">知识库</Link>,
    },
    {
      key: 'messages',
      icon: <MessageOutlined />,
      label: <Link href="/dashboard/messages">消息</Link>,
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: <Link href="/dashboard/settings">设置</Link>,
    },
  ];

  return (
    <Layout className={styles.dashboardLayout}>
      <Header isLoggedIn={true} />
      <Layout className={styles.mainContent}>
        <Sider
          width={240}
          collapsible
          collapsed={collapsed}
          onCollapse={toggleCollapsed}
          className={styles.sider}
          trigger={null}
        >
          <div className={styles.userInfo}>
            <Avatar size={collapsed ? 36 : 64} icon={<UserOutlined />} className={styles.avatar} />
            {!collapsed && (
              <div className={styles.userDetails}>
                <Title level={5} className={styles.userName}>张三</Title>
                <Text className={styles.userRole}>个人用户</Text>
              </div>
            )}
          </div>
          <Button
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={toggleCollapsed}
            className={styles.collapseButton}
          />
          <Menu
            theme="dark"
            mode="inline"
            selectedKeys={[getSelectedKey()]}
            items={menuItems}
            className={styles.siderMenu}
          />
        </Sider>
        <Content className={styles.content}>
          <div className={styles.contentInner}>
            {children}
          </div>
          <Footer />
        </Content>
      </Layout>
    </Layout>
  );
};

export default DashboardLayout; 