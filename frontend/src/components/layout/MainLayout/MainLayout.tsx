import React, { ReactNode } from 'react';
import { Layout } from 'antd';
import Header from '../Header';
import Footer from '../Footer';
import styles from './MainLayout.module.scss';

const { Content } = Layout;

interface MainLayoutProps {
  children: ReactNode;
  isLoggedIn?: boolean;
}

const MainLayout: React.FC<MainLayoutProps> = ({ 
  children, 
  isLoggedIn = false,
}) => {
  return (
    <Layout className={styles.layout}>
      <Header isLoggedIn={isLoggedIn} />
      <Content className={styles.content}>
        <div className={styles.contentInner}>
          {children}
        </div>
      </Content>
      <Footer />
    </Layout>
  );
};

export default MainLayout; 