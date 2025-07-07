import React, { useState } from 'react';
import { Layout, Menu, Button, Avatar, Dropdown, Badge } from 'antd';
import { 
  MenuOutlined, 
  UserOutlined, 
  BellOutlined, 
  MessageOutlined,
  SettingOutlined,
  LogoutOutlined,
  HomeOutlined,
  DesktopOutlined,
  TeamOutlined,
  BookOutlined
} from '@ant-design/icons';
import Link from 'next/link';
import { useRouter } from 'next/router';
import styles from './Header.module.scss';

const { Header: AntHeader } = Layout;

interface HeaderProps {
  isLoggedIn: boolean;
}

const Header: React.FC<HeaderProps> = ({ isLoggedIn }) => {
  const router = useRouter();
  const [mobileMenuVisible, setMobileMenuVisible] = useState(false);

  const toggleMobileMenu = () => {
    setMobileMenuVisible(!mobileMenuVisible);
  };

  const userMenuItems = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: <Link href="/dashboard/profile">个人中心</Link>,
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: <Link href="/dashboard/settings">设置</Link>,
    },
    {
      type: 'divider',
    },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: '退出登录',
      onClick: () => {
        // Handle logout logic
        router.push('/login');
      },
    },
  ];

  const navigationItems = [
    {
      key: 'home',
      icon: <HomeOutlined />,
      label: <Link href="/">首页</Link>,
    },
    {
      key: 'twin',
      icon: <DesktopOutlined />,
      label: <Link href="/dashboard/digital-twin">数字分身</Link>,
    },
    {
      key: 'jobs',
      icon: <TeamOutlined />,
      label: <Link href="/dashboard/jobs">职位</Link>,
    },
    {
      key: 'experts',
      icon: <BookOutlined />,
      label: <Link href="/dashboard/experts">专家</Link>,
    },
  ];

  const getSelectedKey = () => {
    const path = router.pathname;
    if (path === '/') return 'home';
    if (path.includes('/digital-twin')) return 'twin';
    if (path.includes('/jobs')) return 'jobs';
    if (path.includes('/experts')) return 'experts';
    return '';
  };

  return (
    <AntHeader className={styles.header}>
      <div className={styles.logo}>
        <Link href="/">
          <div className={styles.logoContent}>
            <span className={styles.logoText}>数字协同</span>
          </div>
        </Link>
      </div>

      <div className={styles.mobileMenuButton}>
        <Button 
          type="text" 
          icon={<MenuOutlined />} 
          onClick={toggleMobileMenu}
          className={styles.menuButton}
        />
      </div>

      <Menu
        theme="dark"
        mode="horizontal"
        selectedKeys={[getSelectedKey()]}
        className={`${styles.desktopMenu} ${mobileMenuVisible ? styles.mobileMenuVisible : ''}`}
        items={navigationItems}
      />

      {isLoggedIn ? (
        <div className={styles.rightMenu}>
          <Badge count={5} className={styles.notificationBadge}>
            <Button 
              type="text" 
              icon={<BellOutlined />} 
              className={styles.iconButton}
              onClick={() => router.push('/dashboard/notifications')}
            />
          </Badge>
          <Badge count={2} className={styles.messageBadge}>
            <Button 
              type="text" 
              icon={<MessageOutlined />} 
              className={styles.iconButton}
              onClick={() => router.push('/dashboard/messages')}
            />
          </Badge>
          <Dropdown menu={{ items: userMenuItems }} placement="bottomRight">
            <div className={styles.userMenu}>
              <Avatar icon={<UserOutlined />} className={styles.avatar} />
              <span className={styles.username}>用户名</span>
            </div>
          </Dropdown>
        </div>
      ) : (
        <div className={styles.authButtons}>
          <Button 
            type="text" 
            className={styles.loginButton}
            onClick={() => router.push('/login')}
          >
            登录
          </Button>
          <Button 
            type="primary" 
            className={styles.registerButton}
            onClick={() => router.push('/register')}
          >
            注册
          </Button>
        </div>
      )}
    </AntHeader>
  );
};

export default Header; 