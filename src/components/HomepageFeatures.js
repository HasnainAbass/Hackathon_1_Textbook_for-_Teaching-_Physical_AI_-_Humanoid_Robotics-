import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Learn Robotics',
    description: (
      <>
        Comprehensive tutorials on humanoid robotics, from ROS2 fundamentals to advanced AI integration.
      </>
    ),
  },
  {
    title: 'Vision-Language-Action',
    description: (
      <>
        Explore the cutting-edge intersection of computer vision, natural language processing, and robotic action.
      </>
    ),
  },
  {
    title: 'Digital Twins',
    description: (
      <>
        Understand how digital twin technology enables advanced robotics simulation and development.
      </>
    ),
  },
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <div className={styles.featureIcon}>
          <div style={{fontSize: '3rem', lineHeight: '1', marginBottom: '1rem'}}>🤖</div>
        </div>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}